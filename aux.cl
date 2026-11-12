//
//  aux.cl
//  effort_engine_opencl
//
//  Ported from aux.metal
//  Auxiliary kernels for transformer operations
//  FIXED VERSION - No duplicates, proper atomics
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define numHeads 32
#define headDim 128
#define numDims (numHeads * headDim)
#define WG_SIZE 256


inline void atomic_add_float(volatile __global float* addr, float val) {
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

__kernel void touch(__global half* v,
                    __global int* bsScalar,
                    const int vSize) {
    uint id = get_global_id(0);
    uint idx1 = (id * bsScalar[0]) % vSize;
    uint idx2 = (id * bsScalar[0]) % vSize;
    bsScalar[0] += 123 * (int)v[idx1];
    bsScalar[0] += 123 * (int)v[idx2];
    v[idx1] += (half)1e-10;
}

__kernel void hasNan16(__global half* v,
                       volatile __global float* out) {
    uint id = get_global_id(0);
    if (isnan(v[id])) {
        atomic_add_float(out, 1.0f);
    }
}

__kernel void hasNan32(__global float* v,
                       volatile __global float* out) {
    uint id = get_global_id(0);
    if (isnan(v[id])) {
        atomic_add_float(out, 1.0f);
    }
}

__kernel void zero16(__global half* v) {
    uint id = get_global_id(0);
    v[id] = (half)0.0f;
}

__kernel void neg16(__global half* v) {
    uint id = get_global_id(0);
    v[id] = -v[id];
}

__kernel void neg32(__global float* v) {
    uint id = get_global_id(0);
    v[id] = -v[id];
}

__kernel void mulScalar16x16(__global half* v,
                             __global const half* s) {
    uint id = get_global_id(0);
    v[id] = v[id] * s[0];
}

__kernel void mulScalar32x16(__global float* v,
                             __global const half* s) {
    uint id = get_global_id(0);
    v[id] = v[id] * (float)s[0];
}

__kernel void mulScalar16x32(__global half* v,
                             __global const float* s) {
    uint id = get_global_id(0);
    v[id] = (half)(v[id] * s[0]);
}

__kernel void add16(__global const half* v,
                    __global const half* w,
                    __global half* out) {
    uint id = get_global_id(0);
    out[id] = v[id] + w[id];
}

__kernel void strictDiff(__global const half* a,
                         __global const half* b,
                         volatile __global float* diff) {
    uint id = get_global_id(0);
    if (a[id] != b[id]) {
        atomic_add_float(diff, 1.0f);
    }
}

__kernel void rmsNorm32(__global float* input,
                        __global float* output,
                        uint count) {
    
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    
    __local float local_sum[256];
    
    // Each thread computes partial sum with loop unrolling
    float partial = 0.0f;
    for (uint i = lid; i < count; i += lsize) {
        float val = input[i];
        partial += val * val;
    }
    
    local_sum[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction
    #pragma unroll
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Compute RMS and normalize
    float rms = sqrt(local_sum[0] / (float)count + 1e-6f);
    float inv_rms = 1.0f / rms;  // Multiply is faster than divide
    
    output[gid] = input[gid] * inv_rms;
}

// ============================================================================
// rmsNorm32fast - Optimized RMS normalization
// MUST be called with global_size=256, local_size=256
// Each thread processes 16 elements (256 * 16 = 4096)
// ============================================================================
// ============================================================================
// rmsNorm32fast - Optimized RMS normalization
// MUST be called with global_size=256, local_size=256
// Each thread processes 16 elements (256 * 16 = 4096)
// ============================================================================
__kernel void rmsNorm32fast(__global float* input,
                            __global float* output,
                            uint count,
                            __local float* temp) {
    
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);  // 256
    
    // Each thread processes count/lsize elements (4096/256 = 16)
    uint elems_per_thread = count / lsize;
    uint begin = lid * elems_per_thread;
    uint end = begin + elems_per_thread;
    
    // Step 1: Compute partial sum of squares (16 elements per thread)
    float sum = 0.0f;
    for (uint i = begin; i < end; i++) {
        float val = input[i];
        sum += val * val;
    }
    
    // Step 2: Store partial sum for reduction
    temp[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 3: Parallel reduction (log2(256) = 8 iterations)
    for (uint s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            temp[lid] += temp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Step 4: Compute scale factor (all threads read same value)
    float full_sum = temp[0];
    float scale = rsqrt(full_sum / (float)count + 1e-5f);
    
    // Step 5: Normalize output (16 elements per thread)
    for (uint i = begin; i < end; i++) {
        output[i] = input[i] * scale;
    }
}

__kernel void sum_of_exps32(__global const float* input,
                            volatile __global float* sum) {
    uint id = get_global_id(0);
    atomic_add_float(sum, exp(input[id]));
}

__kernel void sum_of_exps32_mx(__global const float* input,
                               volatile __global float* sum,
                               uint numCols) {
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    atomic_add_float(&sum[row], exp(input[col + row * numCols]));
}


__kernel void softmax_add32(__global float* vec,
                            __global float* sum) {
    uint id = get_global_id(0);
    vec[id] = exp(vec[id]) / sum[0];
}

__kernel void softmax_add32_mx(__global float* vec,
                               __global float* sum,
                               uint numCols) {
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    vec[col + row * numCols] = exp(vec[col + row * numCols]) / sum[row];
}

__kernel void rope_mx(__global float* m,
                      __global const float* comp,
                      __global float* out,
                      uint numEls,
                      uint comp_offset) {
    uint id_x = get_global_id(0);  // 0-127 (HEAD_DIM)
    uint id_y = get_global_id(1);  // 0-31 (NUM_HEADS)
    
    uint myOff = id_y * numEls + id_x;
    uint half_dim = numEls / 2;  // 64
    
    uint freq_idx = id_x % half_dim;
    
    float cos_val = comp[comp_offset + freq_idx * 2];
    float sin_val = comp[comp_offset + freq_idx * 2 + 1];
    
    if (id_x < half_dim) {
        out[myOff] = m[myOff] * cos_val - m[myOff + half_dim] * sin_val;
    } else {
        out[myOff] = m[myOff] * cos_val + m[myOff - half_dim] * sin_val;
    }
}

__kernel void mulComplex32_mx(__global float2* m,
                              __global const float2* comp,
                              uint numCols,
                              uint2 id) {
    float a = m[id.x + id.y * numCols / 2].x;
    float b = m[id.x + id.y * numCols / 2].y;
    float c = comp[id.x].x;
    float d = comp[id.x].y;
    
    float2 out;
    out.x = a * c - b * d;
    out.y = a * d + b * c;
    
    m[id.x + id.y * numCols / 2] = out;
}

__kernel void repeat4x32(__global const float* v,
                         __global float* out) {
    uint id_x = get_global_id(0);
    uint id_y = get_global_id(1);
    uint tpg_x = get_global_size(0);
    
    for (int i = 0; i < 4; i++) {
        out[id_x + i * tpg_x + id_y * tpg_x * 4] = v[id_x + id_y * tpg_x];
    }
}

__kernel void compute_attn_scores(__global const float* q_heads,
                                   __global const float* k_cache,
                                   __global float* scores,
                                   uint num_tokens) {
    uint dim_idx = get_global_id(0);
    uint tok_idx = get_global_id(1);
    uint head_idx = get_global_id(2);
    
    if (tok_idx >= num_tokens) return;
    
    float q_val = q_heads[head_idx * headDim + dim_idx];
    float k_val = k_cache[tok_idx * (numHeads * headDim) + head_idx * headDim + dim_idx];
    float contribution = q_val * k_val / sqrt((float)headDim);
    
    atomic_add_float(&scores[head_idx * num_tokens + tok_idx], contribution);
}

__kernel void mulVec16by16(__global const half* v,
                           __global const half* w,
                           __global half* out) {
    uint id = get_global_id(0);
    out[id] = v[id] * w[id];
}

__kernel void mulVec32by16(__global const float* v,
                           __global const half* w,
                           __global float* out) {
    uint id = get_global_id(0);
    out[id] = v[id] * (float)w[id];
}

__kernel void floatToHalf(__global const float* inVec,
                          __global half* outVec) {
    uint id = get_global_id(0);
    outVec[id] = (half)inVec[id];
}

__kernel void halfToFloat(__global const half* inVec,
                          __global float* outVec) {
    uint id = get_global_id(0);
    outVec[id] = (float)inVec[id];
}

__kernel void cosinePrecalc32(__global const float* A,
                              __global const float* B,
                              volatile __global float* dotProduct,
                              volatile __global float* magnitudeA,
                              volatile __global float* magnitudeB) {
    uint id = get_global_id(0);
    atomic_add_float(dotProduct, A[id] * B[id]);
    atomic_add_float(magnitudeA, A[id] * A[id]);
    atomic_add_float(magnitudeB, B[id] * B[id]);
}

__kernel void cosineCalc32(__global float* dotProduct,
                           __global float* magnitudeA,
                           __global float* magnitudeB) {
    dotProduct[0] = dotProduct[0] / (sqrt(magnitudeA[0]) * sqrt(magnitudeB[0]));
}

__kernel void memcpy16(__global const half* src,
                       __global half* dst) {
    uint id = get_global_id(0);
    dst[id] = src[id];
}

__kernel void memcpyBig16(__global const half* src,
                          __global half* dst,
                          uint batchSize) {
    uint id = get_global_id(0);
    uint begin = id * batchSize;
    uint end = begin + batchSize;
    for (uint i = begin; i < end; i++) {
        dst[i] = src[i];
    }
}

__kernel void memcpyBig32(__global const float* src,
                          __global float* dst,
                          uint batchSize) {
    uint id = get_global_id(0);
    uint begin = id * batchSize;
    uint end = begin + batchSize;
    for (uint i = begin; i < end; i++) {
        dst[i] = src[i];
    }
}

__kernel void setVal(__global uint* result,
                     uint number) {
    result[0] = number;
}

__kernel void sumScores32(__global const float* scores,
                          __global const float* xvToken,
                          __global float* out,
                          int numTokens,
                          int scoresCols) {
    uint id = get_global_id(0);
    float suma = 0.0f;
    uint headNo = id / headDim;
    
    for (int tok2 = 0; tok2 < numTokens; tok2++) {
        suma += scores[headNo * scoresCols + tok2] * xvToken[tok2 * numDims + id];
    }
    
    out[id] = suma;
}

__kernel void dotSetScore32(__global const float* v,
                            __global const float* w,
                            __global float* target,
                            int chunkSize,
                            __local float* temp) {
    
    uint id = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    
    float sum = 0.0f;
    uint begin = id * chunkSize;
    uint end = (id + 1) * chunkSize;
    
    for (uint i = begin; i < end; i++) {
        sum += v[i] * w[i];
    }
    
    temp[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            temp[lid] += temp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (id == 0) {
        target[0] = temp[0] / sqrt((float)headDim);
    }
}

__kernel void dot(__global const half* v,
                  __global const half* w,
                  volatile __global float* sum) {
    uint id = get_global_id(0);
    float val = (float)v[id] * (float)w[id];
    atomic_add_float(sum, val);
}

__kernel void setScore(__global const float* sum,
                       __global half* target) {
    target[0] = (half)(sum[0] / sqrt((float)headDim));
}



__kernel void mul16x32(__global const ushort* weights,
                       __global const float* input,
                       __global float* output) {
    int idx = get_global_id(0);
    ushort val = weights[idx];
    output[idx] = convert_float(as_half(val)) * input[idx];
}

__kernel void fetchRow16to32(__global const int* row_idx,
                             __global const ushort* matrix,
                             __global float* output,
                             int cols) {
    int idx = get_global_id(0);
    if (idx >= cols) return;
    
    int row = *row_idx;
    ushort val = matrix[row * cols + idx];
    output[idx] = convert_float(as_half(val));
}

__kernel void matmul16x32(__global const ushort* matrix,
                         __global const float* vector,
                         __global float* output,
                         int rows,
                         int cols,
                         __local float* local_vec,      // ADD THIS
                         __local float* reduce_buf      // ADD THIS
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    
    if (row >= rows) return;
    
    // Cache vector
    for (int i = lid; i < cols; i += WG_SIZE)
        local_vec[i] = vector[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel partial sums
    float partial = 0.0f;
    int base = row * cols;
    for (int i = lid; i < cols; i += WG_SIZE) {
        ushort val = matrix[base + i];
        partial += convert_float(as_half(val)) * local_vec[i];
    }
    
    // Reduction
    reduce_buf[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) reduce_buf[lid] += reduce_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) output[row] = reduce_buf[0];
}

// ============================================================================
// Dot product and score computation - FIXED version
// Removed hardcoded 2048 and 32, removed debug printf
// ============================================================================

__kernel void dotSetScore2(__global const float* xq_heads,
                           __global const float* k_cache, 
                           __global float* scores,
                           int num_tokens,
                           int head_dim,
                           int layer_offset,
                           int num_heads,      // NEW: was hardcoded as 32
                           int max_tokens,     // NEW: was hardcoded as 2048
                           __local float* local_products) {
    
    int dim_idx = get_global_id(0);
    int tok_idx = get_global_id(1);
    int head_idx = get_global_id(2);
    
    int local_dim_idx = get_local_id(0);
    int local_size = get_local_size(0);
    
    if (tok_idx >= num_tokens) return;
    
    // xq_heads layout: [num_heads, head_dim]
    float q_val = xq_heads[head_idx * head_dim + dim_idx];
    
    // k_cache layout: [num_layers, max_tokens, num_heads, head_dim]
    int k_idx = layer_offset + tok_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
    float k_val = k_cache[k_idx];
    
    float partial_product = q_val * k_val;
    
    // Reduction in local memory
    local_products[local_dim_idx] = partial_product;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_dim_idx < stride) {
            local_products[local_dim_idx] += local_products[local_dim_idx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_dim_idx == 0) {
        float score = local_products[0] / sqrt((float)head_dim);
        scores[head_idx * max_tokens + tok_idx] = score;  // FIXED: uses max_tokens
    }
}