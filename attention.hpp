// attention.hpp - PROPERLY FIXED VERSION
// UPDATED: All kernels now tracked via gpu->enqueueKernel()
// Key changes:
// 1. Pre-computed RoPE values for all positions (no runtime sin/cos computation)
// 2. Pre-allocated all temporary buffers (no allocations in forward path)
// 3. NO eval() calls anywhere in this file
// 4. NO blocking writeBuffer/setData calls in forward path
#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "gpu.hpp"
#include "binary_loader.hpp"
#include "bucket_mul.hpp"
#include <memory>
#include <cmath>
#include <fstream>
#include <numeric> 

class Attention {
private:
    static constexpr int NUM_HEADS = 32;
    static constexpr int NUM_KV_HEADS = 8;
    static constexpr int HEAD_DIM = 128;
    static constexpr int HIDDEN_DIM = 4096;
    static constexpr int KV_REPEAT = NUM_HEADS / NUM_KV_HEADS;
    static constexpr int MAX_TOKENS = 2048;
    static constexpr double ROPE_THETA = 1000000.0;
    
    Gpu* gpu;
    BinaryWeightLoader* loader;
    
    // Weight buffers
    BufferWrapper<uint16_t>* wq_buckets;
    BufferWrapper<uint16_t>* wq_stats;
    BufferWrapper<uint16_t>* wq_probes;
    int wq_expert_size;
    int wq_bucket_cols;
    
    BufferWrapper<uint16_t>* wk_buckets;
    BufferWrapper<uint16_t>* wk_stats;
    BufferWrapper<uint16_t>* wk_probes;
    int wk_expert_size;
    int wk_bucket_cols;
    
    BufferWrapper<uint16_t>* wv_buckets;
    BufferWrapper<uint16_t>* wv_stats;
    BufferWrapper<uint16_t>* wv_probes;
    int wv_expert_size;
    int wv_bucket_cols;
    
    BufferWrapper<uint16_t>* wo_buckets;
    BufferWrapper<uint16_t>* wo_stats;
    BufferWrapper<uint16_t>* wo_probes;
    int wo_expert_size;
    int wo_bucket_cols;
    
    // RoPE output buffers
    BufferWrapper<float>* temp_rope_q;
    BufferWrapper<float>* temp_rope_k;
    
    // PRE-ALLOCATED temp buffers (avoid allocation in forward)
    BufferWrapper<float>* temp_xv_repeated;
    BufferWrapper<float>* sum_buf;
    
    // PRE-COMPUTED RoPE values for ALL positions
    // This is the key optimization - we compute sin/cos once at init,
    // not on every forward pass
    std::vector<BufferWrapper<float>*> rope_cache;
    
public:
    Attention(Gpu* g, const std::string& weights_dir, int layer_idx)
        : gpu(g), loader(nullptr) {
        
        loader = new BinaryWeightLoader(gpu, weights_dir);
        std::string layer_prefix = "layer" + std::to_string(layer_idx);
        
        auto load_weights = [&](const std::string& name,
                                BufferWrapper<uint16_t>*& buckets,
                                BufferWrapper<uint16_t>*& stats,
                                BufferWrapper<uint16_t>*& probes,
                                int& expert_size,
                                int& bucket_cols) {
            auto w = loader->load(layer_prefix + "_" + name + "_bucketmul_fp16");
            
            int expert_count = static_cast<int>(w.meta.stats_shape.dim0);
            expert_size = static_cast<int>(w.meta.stats_shape.dim1);
            int stats_channels = static_cast<int>(w.meta.stats_shape.dim2);
            bucket_cols = static_cast<int>(w.meta.buckets_shape.dim2);
            size_t bucket_rows = expert_count * expert_size;
            
            buckets = new BufferWrapper<uint16_t>(gpu, {bucket_rows, (size_t)bucket_cols});
            stats = new BufferWrapper<uint16_t>(gpu, {bucket_rows, (size_t)stats_channels});
            probes = new BufferWrapper<uint16_t>(gpu, {w.meta.probes_shape.dim0, w.meta.probes_shape.dim1});
            
            buckets->setData(gpu, w.buckets_host);  // Blocking OK - init time only
            stats->setData(gpu, w.stats_host);
            probes->setData(gpu, w.probes_host);
        };
        
        load_weights("q_proj", wq_buckets, wq_stats, wq_probes, wq_expert_size, wq_bucket_cols);
        load_weights("k_proj", wk_buckets, wk_stats, wk_probes, wk_expert_size, wk_bucket_cols);
        load_weights("v_proj", wv_buckets, wv_stats, wv_probes, wv_expert_size, wv_bucket_cols);
        load_weights("o_proj", wo_buckets, wo_stats, wo_probes, wo_expert_size, wo_bucket_cols);
        
        // Allocate RoPE output buffers
        temp_rope_q = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        temp_rope_k = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        
        // PRE-ALLOCATE temp buffers
        temp_xv_repeated = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        sum_buf = new BufferWrapper<float>(gpu, NUM_HEADS);
        
        // PRE-COMPUTE RoPE values for all positions
        // This eliminates sin/cos computation and GPU uploads during inference
        precompute_rope_cache();
    }
    
    ~Attention() {
        delete wq_buckets; delete wq_stats; delete wq_probes;
        delete wk_buckets; delete wk_stats; delete wk_probes;
        delete wv_buckets; delete wv_stats; delete wv_probes;
        delete wo_buckets; delete wo_stats; delete wo_probes;
        delete temp_rope_q;
        delete temp_rope_k;
        delete temp_xv_repeated;
        delete sum_buf;
        
        for (auto& buf : rope_cache) {
            delete buf;
        }
        
        delete loader;
    }
    
    void forward(BufferWrapper<float>* input,
                 int layer_idx,
                 int pos,
                 float effort,
                 BufferWrapper<float>* k_cache,
                 BufferWrapper<float>* v_cache,
                 BufferWrapper<float>* temp_xq,
                 BufferWrapper<float>* temp_xk,
                 BufferWrapper<float>* temp_xv,
                 BufferWrapper<float>* temp_xk_repeated,
                 BufferWrapper<float>* temp_xq_heads,
                 BufferWrapper<float>* temp_xk_heads,
                 BufferWrapper<float>* temp_scores,
                 BufferWrapper<float>* temp_attn_output,
                 BufferWrapper<float>* output) {
        
        auto bm = BucketMul::shared(gpu);
        
        // Zero outputs - uses clEnqueueFillBuffer (NON-BLOCKING)
        temp_xq->zero(gpu);
        temp_xk->zero(gpu);
        temp_xv->zero(gpu);
        // NO eval()
        
        // ===== Q/K/V PROJECTIONS =====
        bm->calcDispatch(input, wq_stats, wq_probes, 0, HIDDEN_DIM, wq_bucket_cols, wq_expert_size, effort);
        bm->mul(wq_buckets, temp_xq);
        // NO eval()
        
        bm->calcDispatch(input, wk_stats, wk_probes, 0, HIDDEN_DIM, wk_bucket_cols, wk_expert_size, effort);
        bm->mul(wk_buckets, temp_xk);
        // NO eval()
        
        bm->calcDispatch(input, wv_stats, wv_probes, 0, HIDDEN_DIM, wv_bucket_cols, wv_expert_size, effort);
        bm->mul(wv_buckets, temp_xv);
        // NO eval()
        
        // ===== REPEAT K/V FOR GQA =====
        repeat_kv(temp_xk, temp_xk_repeated);
        repeat_kv(temp_xv, temp_xv_repeated);
        // NO eval()
        
        // ===== RESHAPE TO HEADS =====
        reshape_to_heads(temp_xq, temp_xq_heads);
        reshape_to_heads(temp_xk_repeated, temp_xk_heads);
        // NO eval()
        
        // ===== APPLY ROPE =====
        // Uses PRE-COMPUTED values - NO sin/cos computation, NO setData()!
        apply_rope_precomputed(temp_xq_heads, temp_rope_q, pos);
        apply_rope_precomputed(temp_xk_heads, temp_rope_k, pos);
        // NO eval()
        
        // ===== STORE IN KV CACHE =====
        store_kv_in_cache(temp_rope_k, temp_xv_repeated, layer_idx, pos, k_cache, v_cache);
        // NO eval()
        
        // ===== COMPUTE ATTENTION =====
        int num_tokens = pos + 1;
        compute_attention_with_cache(temp_rope_q, layer_idx, num_tokens, k_cache, v_cache, temp_scores, temp_attn_output);
        // NO eval()
        
        // ===== OUTPUT PROJECTION =====
        output->zero(gpu);  // NON-BLOCKING
        bm->calcDispatch(temp_attn_output, wo_stats, wo_probes, 0, HIDDEN_DIM, wo_bucket_cols, wo_expert_size, effort);
        bm->mul(wo_buckets, output);
        // NO eval() - let model.forward() sync at the very end
    }

private:
    // Pre-compute RoPE sin/cos values for all positions
    // This is called ONCE at initialization, not during inference
    void precompute_rope_cache() {
        std::cout << "    Pre-computing RoPE cache..." << std::flush;
        
        for (int pos = 0; pos < MAX_TOKENS; pos++) {
            std::vector<float> complex_vals(HEAD_DIM * 2);
            
            // Compute frequencies for this position
            for (int i = 0; i < HEAD_DIM / 2; i++) {
                double exponent = (2.0 * i) / HEAD_DIM;
                float freq = 1.0f / std::pow(ROPE_THETA, exponent);
                float angle = pos * freq;
                complex_vals[i * 2] = std::cos(angle);      // real part
                complex_vals[i * 2 + 1] = std::sin(angle);  // imag part
            }
            
            // Second half of HEAD_DIM uses same frequencies
            for (int i = 0; i < HEAD_DIM / 2; i++) {
                complex_vals[(i + 64) * 2] = complex_vals[i * 2];
                complex_vals[(i + 64) * 2 + 1] = complex_vals[i * 2 + 1];
            }
            
            // Upload to GPU - blocking is OK here, this is init time
            auto* buf = new BufferWrapper<float>(gpu, HEAD_DIM * 2);
            buf->setData(gpu, complex_vals);
            rope_cache.push_back(buf);
        }
        
        std::cout << " done (" << rope_cache.size() << " positions)\n";
    }
    
    void reshape_to_heads(BufferWrapper<float>* input, BufferWrapper<float>* output) {
        cl_kernel kernel = gpu->getKernel("memcpy32");
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->buffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->buffer);
        size_t global_size = HIDDEN_DIM / 32;
        
        // TRACKED
        gpu->enqueueKernel1D("memcpy32", kernel, global_size);
        // NO eval()
    }
    
    // Apply RoPE using PRE-COMPUTED values
    // NO sin/cos computation, NO buffer allocation, NO setData()!
    void apply_rope_precomputed(BufferWrapper<float>* heads_in, 
                                 BufferWrapper<float>* heads_out, 
                                 int pos) {
        // Use pre-computed RoPE values from cache
        auto* rope_complex = rope_cache[pos];
        
        cl_kernel kernel = gpu->getKernel("rope_mx");
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &heads_in->buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &rope_complex->buffer);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &heads_out->buffer);
        cl_uint num_els = HEAD_DIM;
        err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &num_els);
        cl_uint comp_offset = 0;
        err |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &comp_offset);
        
        // TRACKED: Use enqueueKernel2D
        gpu->enqueueKernel2D("rope_mx", kernel, HEAD_DIM, NUM_HEADS);
        // NO eval()
    }
    
    void repeat_kv(BufferWrapper<float>* kv_input, BufferWrapper<float>* kv_output) {
        cl_kernel kernel = gpu->getKernel("repeat4x32");
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &kv_input->buffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &kv_output->buffer);
        
        // TRACKED
        gpu->enqueueKernel2D("repeat4x32", kernel, HEAD_DIM, NUM_KV_HEADS);
        // NO eval()
    }
    
    void store_kv_in_cache(BufferWrapper<float>* k_heads, 
                           BufferWrapper<float>* v_vec,
                           int layer_idx,
                           int pos,
                           BufferWrapper<float>* k_cache,
                           BufferWrapper<float>* v_cache) {
        // Copy K to cache - NON-BLOCKING
        size_t layer_k_offset = layer_idx * MAX_TOKENS * NUM_HEADS * HEAD_DIM;
        size_t k_offset = layer_k_offset + pos * NUM_HEADS * HEAD_DIM;
        clEnqueueCopyBuffer(gpu->getQueue(),
                           k_heads->buffer,
                           k_cache->buffer,
                           0,
                           k_offset * sizeof(float),
                           NUM_HEADS * HEAD_DIM * sizeof(float),
                           0, nullptr, nullptr);
        
        // Copy V to cache - NON-BLOCKING
        size_t layer_v_offset = layer_idx * MAX_TOKENS * HIDDEN_DIM;
        size_t v_offset = layer_v_offset + pos * HIDDEN_DIM;
        clEnqueueCopyBuffer(gpu->getQueue(),
                           v_vec->buffer,
                           v_cache->buffer,
                           0,
                           v_offset * sizeof(float),
                           HIDDEN_DIM * sizeof(float),
                           0, nullptr, nullptr);
        // NO eval()
    }
    
    void compute_attention_with_cache(BufferWrapper<float>* q_heads,
                                      int layer_idx,
                                      int num_tokens,
                                      BufferWrapper<float>* k_cache,
                                      BufferWrapper<float>* v_cache,
                                      BufferWrapper<float>* scores,
                                      BufferWrapper<float>* output) {
        
        // Zero scores - NON-BLOCKING
        scores->zero(gpu);
        // NO eval()
        
        size_t layer_k_offset = layer_idx * MAX_TOKENS * NUM_HEADS * HEAD_DIM;
        
        // ===== COMPUTE ATTENTION SCORES =====
        cl_kernel score_kernel = gpu->getKernel("dotSetScore2");
        cl_int err;
        err = clSetKernelArg(score_kernel, 0, sizeof(cl_mem), &q_heads->buffer);
        cl_mem k_cache_buf = k_cache->buffer;
        err |= clSetKernelArg(score_kernel, 1, sizeof(cl_mem), &k_cache_buf);
        err |= clSetKernelArg(score_kernel, 2, sizeof(cl_mem), &scores->buffer);
        cl_int nt = num_tokens;
        err |= clSetKernelArg(score_kernel, 3, sizeof(cl_int), &nt);
        cl_int hd = HEAD_DIM;
        err |= clSetKernelArg(score_kernel, 4, sizeof(cl_int), &hd);
        cl_int layer_offset = (cl_int)layer_k_offset;
        err |= clSetKernelArg(score_kernel, 5, sizeof(cl_int), &layer_offset);
        cl_int nh = NUM_HEADS;
        err |= clSetKernelArg(score_kernel, 6, sizeof(cl_int), &nh);
        cl_int mt = MAX_TOKENS;
        err |= clSetKernelArg(score_kernel, 7, sizeof(cl_int), &mt);
        err |= clSetKernelArg(score_kernel, 8, HEAD_DIM * sizeof(float), nullptr);
        
        size_t global[3] = {(size_t)HEAD_DIM, (size_t)num_tokens, (size_t)NUM_HEADS};
        size_t local[3] = {(size_t)HEAD_DIM, 1, 1};
        
        // TRACKED
        gpu->enqueueKernel("dotSetScore2", score_kernel, 3, nullptr, global, local);
        // NO eval()
        
        // ===== SOFTMAX: SUM OF EXPS =====
        sum_buf->zero(gpu);  // NON-BLOCKING
        
        cl_kernel sum_kernel = gpu->getKernel("sum_of_exps32_mx");
        cl_mem scores_buf = scores->buffer;
        err = clSetKernelArg(sum_kernel, 0, sizeof(cl_mem), &scores_buf);
        err |= clSetKernelArg(sum_kernel, 1, sizeof(cl_mem), &sum_buf->buffer);
        cl_uint num_cols = MAX_TOKENS;
        err |= clSetKernelArg(sum_kernel, 2, sizeof(cl_uint), &num_cols);
        
        // TRACKED
        gpu->enqueueKernel2D("sum_of_exps32_mx", sum_kernel, num_tokens, NUM_HEADS);
        // NO eval() - OpenCL in-order queue handles dependency!
        
        // ===== SOFTMAX: DIVIDE =====
        cl_kernel softmax_kernel = gpu->getKernel("softmax_add32_mx");
        err = clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &scores_buf);
        err |= clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &sum_buf->buffer);
        err |= clSetKernelArg(softmax_kernel, 2, sizeof(cl_uint), &num_cols);
        
        // TRACKED
        gpu->enqueueKernel2D("softmax_add32_mx", softmax_kernel, num_tokens, NUM_HEADS);
        // NO eval()
        
        // ===== WEIGHTED SUM OF VALUES =====
        output->zero(gpu);  // NON-BLOCKING
        
        size_t layer_v_offset = layer_idx * MAX_TOKENS * HIDDEN_DIM;
        cl_buffer_region region;
        region.origin = layer_v_offset * sizeof(float);
        region.size = MAX_TOKENS * HIDDEN_DIM * sizeof(float);
        
        cl_mem v_layer_buf = clCreateSubBuffer(
            v_cache->buffer,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &err
        );
        
        cl_kernel sumScores_kernel = gpu->getKernel("sumScores32");
        err = clSetKernelArg(sumScores_kernel, 0, sizeof(cl_mem), &scores_buf);
        err |= clSetKernelArg(sumScores_kernel, 1, sizeof(cl_mem), &v_layer_buf);
        err |= clSetKernelArg(sumScores_kernel, 2, sizeof(cl_mem), &output->buffer);
        err |= clSetKernelArg(sumScores_kernel, 3, sizeof(cl_int), &nt);
        cl_int score_cols = MAX_TOKENS;
        err |= clSetKernelArg(sumScores_kernel, 4, sizeof(cl_int), &score_cols);
        
        // TRACKED
        gpu->enqueueKernel1D("sumScores32", sumScores_kernel, HIDDEN_DIM);
        // NO eval()
        
        clReleaseMemObject(v_layer_buf);
    }
};

#endif