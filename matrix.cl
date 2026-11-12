//
//  matrix.cl
//  effort_engine_opencl
//
//  Ported from matrix.metal
//  Matrix operations for effort control
//  FIXED VERSION - No findCutoff32 (it's in bucketMul.cl)
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Note: findCutoff32 is in bucketMul.cl to avoid duplication

// ============================================================================
// SiLU activation function
// ============================================================================

__kernel void silu(__global const half* x1,
                   __global const half* x3,
                   __global half* out) {
    uint id = get_global_id(0);
    float x1_f = (float)x1[id];
    float x3_f = (float)x3[id];
    out[id] = (half)(x3_f * x1_f / (1.0f + exp(-x1_f)));
}

__kernel void silu32(__global const float* x1,
                     __global const float* x3,
                     __global float* out) {
    uint id = get_global_id(0);
    out[id] = x3[id] * x1[id] / (1.0f + exp(-x1[id]));
}

__kernel void silu32b(__global const float* x1,
                      __global const float* x3,
                      __global float* out) {
    uint id = get_global_id(0);
    uint begin = id * 64;
    uint end = id * 64 + 64;
    for (uint i = begin; i < end; i++) {
        out[i] = x3[i] * x1[i] / (1.0f + exp(-x1[i]));
    }
}

// ============================================================================
// Simplified findCutoff (half precision) - for reference only
// Real findCutoff32 is in bucketMul.cl
// ============================================================================

__kernel void findCutoff(__global const float* v,
                         __global const half* probes,
                         __global const uint* expNo,
                         __global half* out,
                         uint _effort,
                         uint id) {
    
    // Simplified version - just computes average
    if (id == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 4096; i++) {
            sum += fabs(v[i] * (float)probes[i + expNo[0] * 4096]);
        }
        out[0] = (half)(sum / 4096.0f);
    }
}

// ============================================================================
// Matrix operations
// ============================================================================

__kernel void getVal(__global const half* vector,
                     __global half* val,
                     uint pos,
                     uint id) {
    if (id == pos) {
        val[0] = vector[pos];
    }
}

__kernel void basicMul(__global const float* v,
                       __global const half* m,
                       __global float* out,
                       uint numCols,
                       uint rowId) {
    float sum = 0.0f;
    uint offset = rowId * numCols;
    
    for (uint i = 0; i < numCols; i++) {
        sum += v[i] * (float)m[i + offset];
    }
    
    out[rowId] = sum;
}

// Helper for atomic add on float
inline void atomic_add_float_global(volatile __global float* addr, float val) {
    union {
        float f;
        uint i;
    } old_val;
    union {
        float f;
        uint i;
    } new_val;
    
    do {
        old_val.f = *addr;
        new_val.f = old_val.f + val;
    } while (atomic_cmpxchg((volatile __global uint*)addr, old_val.i, new_val.i) != old_val.i);
}

__kernel void basicMul2(__global const float* v,
                        __global const half* m,
                        volatile __global float* out,
                        uint numCols,
                        uint chunkSize,
                        uint2 rowId) {
    float sum = 0.0f;
    uint offset = rowId.x * numCols + rowId.y * chunkSize;
    
    for (uint i = 0; i < chunkSize; i++) {
        sum += v[rowId.y * chunkSize + i] * (float)m[i + offset];
    }
    
    atomic_add_float_global(&out[rowId.x], sum);
}

// ============================================================================
// Bitonic sort (for sorting values)
// ============================================================================

__kernel void basicBitonicSort(__global half* floats,
                               int p,
                               int q,
                               uint gid) {
    // Bitonic sort implementation
    // Taken from Apple's Metal sort example
    
    int pMinusQ = p - q;
    int distance = 1 << pMinusQ;
    uint gidShiftedByP = gid >> p;
    
    // True: Increasing / False: Decreasing
    bool direction = ((gidShiftedByP & 2) == 0);
    uint gidDistance = (gid & distance);
    bool isGidDistanceZero = (gidDistance == 0);
    uint gidPlusDistance = (gid | distance);
    
    bool isLowerIndexGreaterThanHigher = (floats[gid] > floats[gidPlusDistance]);
    
    if (isGidDistanceZero && isLowerIndexGreaterThanHigher == direction) {
        half temp = floats[gid];
        floats[gid] = floats[gidPlusDistance];
        floats[gidPlusDistance] = temp;
    }
}