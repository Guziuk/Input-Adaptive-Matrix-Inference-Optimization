//
//  bucketMul_largecache.cl - SIMPLE OPTIMIZATION
//  
//  Just increase DISPATCH_CACHE_SIZE from 64 to 256
//  This amortizes global memory latency better without adding kernel overhead
//
//  RATIONALE: 
//  - Larger LDS cache = more dispatch entries fetched per barrier
//  - Fewer global memory round-trips
//  - No extra kernel launches (unlike binning)
//
//  LDS usage: 256 entries × 8 bytes = 2KB per workgroup (well within 64KB limit)
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define CUTOFF_SCALE 100000.0f
#define PROBES_COUNT 4096
#define NUM_BINS 256
#define WORKGROUP_SIZE 256
#define VALUES_PER_THREAD 16
#define SUBGROUP_SIZE 32
#define NUM_SUBGROUPS 8
#define THREADS_PER_GROUP 256
// INCREASED from 64 to 256
#define DISPATCH_CACHE_SIZE 256
#define SPARSE_THRESHOLD 16384 


inline float warp_reduce_min(float val) {
    // AMD wavefront is 64 threads
    val = min(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 1));
    val = min(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 2));
    val = min(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 4));
    val = min(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 8));
    val = min(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 16));
    val = min(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 32));
    return val;
}



// Warp/wavefront reduction for max
inline float warp_reduce_max(float val) {
    val = max(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 1));
    val = max(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 2));
    val = max(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 4));
    val = max(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 8));
    val = max(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 16));
    val = max(val, sub_group_broadcast(val, get_sub_group_local_id() ^ 32));
    return val;
}

// Warp/wavefront reduction for sum (int)
inline int warp_reduce_sum_int(int val) {
    val += sub_group_broadcast(val, get_sub_group_local_id() ^ 1);
    val += sub_group_broadcast(val, get_sub_group_local_id() ^ 2);
    val += sub_group_broadcast(val, get_sub_group_local_id() ^ 4);
    val += sub_group_broadcast(val, get_sub_group_local_id() ^ 8);
    val += sub_group_broadcast(val, get_sub_group_local_id() ^ 16);
    val += sub_group_broadcast(val, get_sub_group_local_id() ^ 32);
    return val;
}




__kernel void findCutoffBinarySearch(
    __global const float *v,
    __global const half *probes,
    __global const int *expNo,
    __global float *out,
    const int _effort,
    __local float *shared_mem)  // Size: 1024 floats (4KB)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint warp_id = lid / 64;           // Which wavefront (0-3)
    const uint lane_id = lid % 64;           // Position in wavefront (0-63)
    
    const int effort = PROBES_COUNT - _effort;
    
    // Shared memory layout:
    // [0-3]:   warp_mins
    // [4-7]:   warp_maxs
    // [8-11]:  warp_counts
    // [12-15]: bounds (minBound, maxBound, newBound, done_flag)
    // [16-17]: counts (minCount, maxCount)
    
    __local float *warp_mins = shared_mem;
    __local float *warp_maxs = shared_mem + 4;
    __local int *warp_counts = (__local int *)(shared_mem + 8);
    __local float *bounds = shared_mem + 12;
    __local int *counts = (__local int *)(shared_mem + 16);
    
    // Each thread holds 16 importance values
    float myVals[VALUES_PER_THREAD];
    float myMin = 1e30f;
    float myMax = -1e30f;
    
    // Load and compute importance values
    uint base_idx = gid * VALUES_PER_THREAD;
    int probe_offset = expNo[0] * PROBES_COUNT;
    
    #pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
        int idx = base_idx + i;
        if (idx < PROBES_COUNT) {
            float probe_val = vload_half(idx + probe_offset, probes);
            float importance = CUTOFF_SCALE * fabs(v[idx] * probe_val);
            myVals[i] = importance;
            myMin = min(myMin, importance);
            myMax = max(myMax, importance);
        } else {
            myVals[i] = 0.0f;
        }
    }
    
    // Reduce min/max within each wavefront
    float warp_min = warp_reduce_min(myMin);
    float warp_max = warp_reduce_max(myMax);
    
    // First thread in each wavefront stores result
    if (lane_id == 0) {
        warp_mins[warp_id] = warp_min;
        warp_maxs[warp_id] = warp_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Thread 0 finds global min/max and initializes search
    if (lid == 0) {
        float global_min = min(min(warp_mins[0], warp_mins[1]), 
                              min(warp_mins[2], warp_mins[3]));
        float global_max = max(max(warp_maxs[0], warp_maxs[1]), 
                              max(warp_maxs[2], warp_maxs[3]));
        
        bounds[0] = global_min;      // minBound
        bounds[1] = global_max;      // maxBound
        bounds[2] = (global_min + global_max) / 2.0f;  // newBound
        bounds[3] = 0.0f;            // done_flag
        
        counts[0] = PROBES_COUNT;    // minCount (all values > global_min)
        counts[1] = 0;               // maxCount (no values > global_max)
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Binary search loop
    for (int loop = 0; loop < 20; loop++) {
        float newBound = bounds[2];
        
        // Count how many of my values are above newBound
        int myAbove = 0;
        #pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; i++) {
            myAbove += (myVals[i] > newBound) ? 1 : 0;
        }
        
        // Reduce count within wavefront
        int warp_sum = warp_reduce_sum_int(myAbove);
        
        // First thread in each wavefront stores result
        if (lane_id == 0) {
            warp_counts[warp_id] = warp_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Thread 0 updates search bounds
        if (lid == 0) {
            int globalCount = warp_counts[0] + warp_counts[1] + 
                             warp_counts[2] + warp_counts[3];
            
            float minBound = bounds[0];
            float maxBound = bounds[1];
            int minCount = counts[0];
            int maxCount = counts[1];
            
            // Update bounds based on count
            if (globalCount < effort) {
                // Too few selected, lower the threshold
                maxBound = newBound;
                maxCount = globalCount;
            } else {
                // Too many or just right, raise the threshold
                minBound = newBound;
                minCount = globalCount;
            }
            
            bounds[0] = minBound;
            bounds[1] = maxBound;
            bounds[2] = (minBound + maxBound) / 2.0f;
            counts[0] = minCount;
            counts[1] = maxCount;
            
            // Check convergence
            if (globalCount == effort ||                    // Exact match
                (maxBound - minBound < 0.00001f) ||        // Converged
                abs(maxCount - minCount) < 3) {            // Close enough
                out[0] = newBound;
                bounds[3] = 1.0f;  // Signal done
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Check if we're done
        if (bounds[3] > 0.5f) {
            return;
        }
    }
    
    // If we hit max iterations, use final bound
    if (lid == 0) {
        out[0] = bounds[2];
    }
}



// ============================================================================
// findCutoff32 - unchanged
// ============================================================================
__kernel void findCutoffBinarySearch_fallbackaa(
    __global const float *v,
    __global const half *probes,
    __global const int *expNo,
    __global float *out,
    const int _effort,
    __local float *shared_mem)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    
    const int effort = PROBES_COUNT - _effort;
    
    // Shared memory layout (same as above)
    __local float *thread_mins = shared_mem;           // [0-255]
    __local float *thread_maxs = shared_mem + 256;     // [256-511]
    __local int *thread_counts = (__local int *)(shared_mem + 512);  // [512-767]
    __local float *bounds = shared_mem + 768;          // [768-771]
    __local int *counts = (__local int *)(shared_mem + 772);         // [772-773]
    
    float myVals[VALUES_PER_THREAD];
    float myMin = 1e30f;
    float myMax = -1e30f;
    
    uint base_idx = gid * VALUES_PER_THREAD;
    int probe_offset = expNo[0] * PROBES_COUNT;
    
    #pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
        int idx = base_idx + i;
        if (idx < PROBES_COUNT) {
            float probe_val = vload_half(idx + probe_offset, probes);
            float importance = CUTOFF_SCALE * fabs(v[idx] * probe_val);
            myVals[i] = importance;
            myMin = min(myMin, importance);
            myMax = max(myMax, importance);
        } else {
            myVals[i] = 0.0f;
        }
    }
    
    // Store my min/max
    thread_mins[lid] = myMin;
    thread_maxs[lid] = myMax;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction for global min/max
    for (uint s = WORKGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            thread_mins[lid] = min(thread_mins[lid], thread_mins[lid + s]);
            thread_maxs[lid] = max(thread_maxs[lid], thread_maxs[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        bounds[0] = thread_mins[0];  // minBound
        bounds[1] = thread_maxs[0];  // maxBound
        bounds[2] = (thread_mins[0] + thread_maxs[0]) / 2.0f;  // newBound
        bounds[3] = 0.0f;            // done_flag
        counts[0] = PROBES_COUNT;
        counts[1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Binary search loop
    for (int loop = 0; loop < 20; loop++) {
        float newBound = bounds[2];
        
        int myAbove = 0;
        #pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; i++) {
            myAbove += (myVals[i] > newBound) ? 1 : 0;
        }
        
        thread_counts[lid] = myAbove;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Tree reduction for global count
        for (uint s = WORKGROUP_SIZE / 2; s > 0; s >>= 1) {
            if (lid < s) {
                thread_counts[lid] += thread_counts[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (lid == 0) {
            int globalCount = thread_counts[0];
            float minBound = bounds[0];
            float maxBound = bounds[1];
            int minCount = counts[0];
            int maxCount = counts[1];
            
            if (globalCount < effort) {
                maxBound = newBound;
                maxCount = globalCount;
            } else {
                minBound = newBound;
                minCount = globalCount;
            }
            
            bounds[0] = minBound;
            bounds[1] = maxBound;
            bounds[2] = (minBound + maxBound) / 2.0f;
            counts[0] = minCount;
            counts[1] = maxCount;
            
            if (globalCount == effort || 
                (maxBound - minBound < 0.00001f) ||
                abs(maxCount - minCount) < 3) {
                out[0] = newBound;
                bounds[3] = 1.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (bounds[3] > 0.5f) {
            return;
        }
    }
    
    if (lid == 0) {
        out[0] = bounds[2];
    }
}

#define BATCH_SIZE 8
// ============================================================================
// prepareDispatch - unchanged
// ============================================================================
__kernel void prepareDispatch(__global const float* v,
                              __global const half* stats,
                              __global const int* expertNo,
                              __global const float* cutoff,
                              __global float2* dispatch,
                              __global atomic_uint* dispatchCount,
                              const int chunkSize,
                              const int inSize,
                              const int bucketCols,
                              const int expertSize) {
    
    uint id = get_global_id(0);
    uint dispatchOffset = expertSize * (*expertNo);
    uint begin = chunkSize * id + dispatchOffset;
    uint end = begin + chunkSize;
    
    if (end > dispatchOffset + expertSize) end = dispatchOffset + expertSize;
    
    float cutoffVal = *cutoff;
    
    for (uint i = begin; i < end; i++) {
        float avg = (float)stats[i * 4 + 3];
        uint bucketRow = i - dispatchOffset;
        uint inputIdx = bucketRow / 16;
        
        if (inputIdx < (uint)inSize) {
            float val = v[inputIdx];
            float importance = CUTOFF_SCALE * fabs(val * avg);
            
            if (importance >= cutoffVal) {
                uint dispatchIdx = atomic_fetch_add(dispatchCount, 1);
                dispatch[dispatchIdx] = (float2)(val, (float)(i * bucketCols));
            }
        }
    }
}

// ============================================================================
// bucketMul - with LARGER LDS cache (256 vs 64)
// ============================================================================

__kernel void bucketMul(__global const half* weights,
                        __global const float2* dispatch,
                        __global float* result,
                        __global const uint* dispatchSize,
                        const int cols,
                        const int groups,
                        __local float2* dispatchCache)
{
    uint lid = get_local_id(0);
    uint wgSize = get_local_size(0);
    uint col = get_group_id(0) * wgSize + lid;
    uint groupId = get_global_id(1);
    
    if (col >= cols) return;
    
    float myVal[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) myVal[i] = 0.0f;
    
    uint dispSize = *dispatchSize;
    uint chunkSize = dispSize / groups;
    uint rowOffset = groupId * chunkSize;
    uint rowEnd = rowOffset + chunkSize;
    
    for (uint chunkStart = rowOffset; chunkStart < rowEnd; chunkStart += DISPATCH_CACHE_SIZE) {
        uint chunkEnd = min(chunkStart + DISPATCH_CACHE_SIZE, rowEnd);
        uint chunkLen = chunkEnd - chunkStart;
        
        for (uint i = lid; i < chunkLen; i += wgSize) {
            dispatchCache[i] = dispatch[chunkStart + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (uint r = 0; r < chunkLen; r++) {
            float2 d = dispatchCache[r];
            float inputVal = d.x;
            int weightOffset = (int)d.y;
            
            half w = weights[weightOffset + col];
            float product = inputVal * (float)w;
            uint pos = as_ushort(w) & 0xF;
            
            myVal[pos] += product;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // KEY CHANGE: Write in TRANSPOSED layout for coalesced reads
    // Old: result[groupId * 16384 + col * 16 + i]
    // New: result[col * 16 * groups + i * groups + groupId]
    uint baseIdx = col * 16 * groups;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[baseIdx + i * groups + groupId] = myVal[i];
    }
}

// ============================================================================
// Rest unchanged
// ============================================================================

__kernel void bucketIntegrate(__global const float* tmpMulVec,
                              __global float* out,
                              __local float* scratch)  // Use this!
{
    uint localId = get_local_id(0);
    uint groupId = get_group_id(1);
    const uint INTSTEPS = 4;
    const uint THREADS_PER_OUTPUT = 16;
    const uint MUL_GROUPS = 64;
    
    uint outputBase = groupId * INTSTEPS;
    uint outputNum = localId / THREADS_PER_OUTPUT;     // 0-3 (which output)
    uint threadInOutput = localId % THREADS_PER_OUTPUT; // 0-15 (position in output)
    
    uint outputIdx = outputBase + outputNum;
    
    // Each thread accumulates 4 values
    float mySum = 0.0f;
    uint baseAddr = outputIdx * MUL_GROUPS;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        mySum += tmpMulVec[baseAddr + threadInOutput + i * THREADS_PER_OUTPUT];
    }
    
    // Store to shared memory - 4 groups of 16 threads each
    uint subgroup_base = outputNum * THREADS_PER_OUTPUT;
    scratch[subgroup_base + threadInOutput] = mySum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction within each group of 16 threads
    // 16 → 8 → 4 → 2 → 1 (4 barrier stages)
    for (uint s = 8; s > 0; s >>= 1) {
        if (threadInOutput < s) {
            scratch[subgroup_base + threadInOutput] += 
                scratch[subgroup_base + threadInOutput + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // First thread in each subgroup writes result
    if (threadInOutput == 0) {
        out[outputIdx] = scratch[subgroup_base];
    }
}


__kernel void roundUp(__global atomic_uint* size, __global uint* prevSize, const int roundTo) {
    if (get_global_id(0) == 0) {
        uint currentSize = atomic_load(size);
        uint rounded = ((currentSize + roundTo - 1) / roundTo) * roundTo;
        *prevSize = currentSize;
        atomic_store(size, rounded);
    }
}

__kernel void zeroRange32(__global float2* buffer, __global const uint* begin, __global const uint* end) {
    uint id = get_global_id(0);
    uint idx = *begin + id;
    if (idx < *end) buffer[idx] = (float2)(0.0f, 0.0f);
}

__kernel void zero32(__global float* v) { v[get_global_id(0)] = 0.0f; }

__kernel void add32(__global const float* v, __global const float* w, __global float* out) {
    uint id = get_global_id(0);
    out[id] = v[id] + w[id];
}

__kernel void memcpy32(__global const float* src, __global float* dst) {
    uint id = get_global_id(0);
    uint base = id * 32;
    
    __global const float4* src4 = (__global const float4*)(src + base);
    __global float4* dst4 = (__global float4*)(dst + base);
    
    // Copy 8 float4s = 32 floats
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst4[i] = src4[i];
    }
}

__kernel void calcOutliers(__global const float* v, __global const half* outliers, __global float* out) {
    uint id = get_global_id(0);
    out[id] += v[id] * (float)outliers[id];
}

