// model.hpp - FIXED VERSION - Only ONE eval() at the end
// UPDATED: All kernels now tracked via gpu->enqueueKernel()
#ifndef MODEL_HPP
#define MODEL_HPP

#include "gpu.hpp"
#include "transformer_layer.hpp"
#include "binary_loader.hpp"
#include "bucket_mul.hpp"
#include <vector>
#include <memory>
#include <string>
#include <fstream>

class MistralModel {
private:
    Gpu* gpu;
    
    BufferWrapper<uint16_t>* token_embeddings;
    BufferWrapper<uint16_t>* final_norm_weight;
    BufferWrapper<uint16_t>* lm_head;
    
    std::vector<std::unique_ptr<TransformerLayer>> layers;
    
    static constexpr int VOCAB_SIZE = 32768;
    static constexpr int HIDDEN_DIM = 4096;
    static constexpr int NUM_LAYERS = 32;
    static constexpr int NUM_HEADS = 32;
    static constexpr int HEAD_DIM = 128;
    static constexpr int MAX_TOKENS = 2048;
    
    BufferWrapper<float>* k_cache;
    BufferWrapper<float>* v_cache;
    
    BufferWrapper<float>* h_norm;
    BufferWrapper<float>* temp_xq;
    BufferWrapper<float>* temp_xk;
    BufferWrapper<float>* temp_xv;
    BufferWrapper<float>* temp_xk_repeated;
    BufferWrapper<float>* temp_xq_heads;
    BufferWrapper<float>* temp_xk_heads;
    BufferWrapper<float>* temp_scores;
    BufferWrapper<float>* temp_attn_output;
    BufferWrapper<float>* attn_out;
    BufferWrapper<float>* fxn;
    BufferWrapper<float>* ffn_out;
    
    BufferWrapper<float>* hidden_states;
    
    // PRE-ALLOCATED buffers (moved from forward() to avoid allocation in hot path)
    BufferWrapper<int>* token_id_buf;
    BufferWrapper<float>* final_normed;
    
    std::vector<uint16_t> load_matrix_fp16(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open: " + filepath);
        }
        
        char magic[6];
        file.read(magic, 6);
        
        uint8_t major, minor;
        file.read((char*)&major, 1);
        file.read((char*)&minor, 1);
        
        uint16_t header_len;
        file.read((char*)&header_len, 2);
        
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(10 + header_len, std::ios::beg);
        
        size_t data_size = file_size - (10 + header_len);
        size_t num_elements = data_size / sizeof(uint16_t);
        
        std::vector<uint16_t> data(num_elements);
        file.read((char*)data.data(), data_size);
        
        return data;
    }
    
public:
    MistralModel(Gpu* g, const std::string& weights_dir) : gpu(g) {
        auto embed = load_matrix_fp16(weights_dir + "/embed_tokens.npy");
        token_embeddings = new BufferWrapper<uint16_t>(gpu, VOCAB_SIZE * HIDDEN_DIM);
        token_embeddings->setData(gpu, embed);
        
        auto norm = load_matrix_fp16(weights_dir + "/norm_weight.npy");
        final_norm_weight = new BufferWrapper<uint16_t>(gpu, HIDDEN_DIM);
        final_norm_weight->setData(gpu, norm);
        
        auto lm = load_matrix_fp16(weights_dir + "/lm_head.npy");
        lm_head = new BufferWrapper<uint16_t>(gpu, VOCAB_SIZE * HIDDEN_DIM);
        lm_head->setData(gpu, lm);
        
        k_cache = new BufferWrapper<float>(gpu, {(size_t)NUM_LAYERS, (size_t)MAX_TOKENS, (size_t)NUM_HEADS, (size_t)HEAD_DIM});
        v_cache = new BufferWrapper<float>(gpu, {(size_t)NUM_LAYERS, (size_t)MAX_TOKENS, (size_t)HIDDEN_DIM});
        
        h_norm = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        temp_xq = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        temp_xk = new BufferWrapper<float>(gpu, NUM_HEADS / 4 * HEAD_DIM);
        temp_xv = new BufferWrapper<float>(gpu, NUM_HEADS / 4 * HEAD_DIM);
        temp_xk_repeated = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        temp_xq_heads = new BufferWrapper<float>(gpu, {(size_t)NUM_HEADS, (size_t)HEAD_DIM});
        temp_xk_heads = new BufferWrapper<float>(gpu, {(size_t)NUM_HEADS, (size_t)HEAD_DIM});
        temp_scores = new BufferWrapper<float>(gpu, {(size_t)NUM_HEADS, (size_t)MAX_TOKENS});
        temp_attn_output = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        attn_out = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        fxn = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        ffn_out = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        hidden_states = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        
        // PRE-ALLOCATE buffers that were being created in forward()
        token_id_buf = new BufferWrapper<int>(gpu, 1);
        final_normed = new BufferWrapper<float>(gpu, HIDDEN_DIM);
        
        std::cout << "Loading layers...\n";
        for (int i = 0; i < NUM_LAYERS; i++) {
            std::cout << "  Loading layer " << i << "..." << std::flush;
            layers.push_back(std::make_unique<TransformerLayer>(gpu, weights_dir, i));
            std::cout << " done\n";
        }
        
        std::cout << "Model loaded (" << NUM_LAYERS << " layers)\n";
        gpu->memTracker->print_status();
    }
    
    ~MistralModel() {
        delete token_embeddings;
        delete final_norm_weight;
        delete lm_head;
        delete k_cache;
        delete v_cache;
        delete h_norm;
        delete temp_xq;
        delete temp_xk;
        delete temp_xv;
        delete temp_xk_repeated;
        delete temp_xq_heads;
        delete temp_xk_heads;
        delete temp_scores;
        delete temp_attn_output;
        delete attn_out;
        delete fxn;
        delete ffn_out;
        delete hidden_states;
        delete token_id_buf;
        delete final_normed;
    }
    
    void forward(int token_id, int pos, float effort, BufferWrapper<float>* logits_out) {
    
    // ===== EMBEDDING LOOKUP =====
    std::vector<int> token_vec = {token_id};
    token_id_buf->setData(gpu, token_vec);
    
    cl_kernel fetch_kernel = gpu->getKernel("fetchRow16to32");
    clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), &token_id_buf->buffer);
    clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), &token_embeddings->buffer);
    clSetKernelArg(fetch_kernel, 2, sizeof(cl_mem), &hidden_states->buffer);
    int cols = HIDDEN_DIM;
    clSetKernelArg(fetch_kernel, 3, sizeof(int), &cols);
    
    // TRACKED: Use enqueueKernel1D instead of raw clEnqueueNDRangeKernel
    gpu->enqueueKernel1D("fetchRow16to32", fetch_kernel, HIDDEN_DIM);
    // NO eval() here!
    
    // ===== RUN ALL TRANSFORMER LAYERS =====
    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++) {
        layers[layer_idx]->forward(hidden_states, layer_idx, pos, effort,
                                  k_cache, v_cache,
                                  h_norm, temp_xq, temp_xk, temp_xv, temp_xk_repeated,
                                  temp_xq_heads, temp_xk_heads, temp_scores, temp_attn_output,
                                  attn_out, fxn, ffn_out,
                                  hidden_states);
        // NO eval() here - layers run back-to-back!
    }
    
    // ===== FINAL RMS NORM =====
    cl_kernel norm_kernel = gpu->getKernel("rmsNorm32fast");
    cl_int err;
    err = clSetKernelArg(norm_kernel, 0, sizeof(cl_mem), &hidden_states->buffer);
    err |= clSetKernelArg(norm_kernel, 1, sizeof(cl_mem), &final_normed->buffer);
    cl_uint count = HIDDEN_DIM;
    err |= clSetKernelArg(norm_kernel, 2, sizeof(cl_uint), &count);
    err |= clSetKernelArg(norm_kernel, 3, 256 * sizeof(float), nullptr);

    size_t global_size = 256;
    size_t local_size = 256;
    
    // TRACKED: Use enqueueKernel instead of raw clEnqueueNDRangeKernel
    gpu->enqueueKernel("rmsNorm32fast", norm_kernel, 1, nullptr, &global_size, &local_size);
    // NO eval() here!

    // ===== MULTIPLY BY NORM WEIGHTS =====
    cl_kernel mul_kernel = gpu->getKernel("mulVec32by16");
    clSetKernelArg(mul_kernel, 0, sizeof(cl_mem), &final_normed->buffer);
    clSetKernelArg(mul_kernel, 1, sizeof(cl_mem), &final_norm_weight->buffer);
    clSetKernelArg(mul_kernel, 2, sizeof(cl_mem), &final_normed->buffer);
    
    // TRACKED
    gpu->enqueueKernel1D("mulVec32by16", mul_kernel, HIDDEN_DIM);
    // NO eval() here!
    
    // ===== LM HEAD PROJECTION (OPTIMIZED) =====
    logits_out->zero(gpu);
    
    cl_kernel lm_kernel = gpu->getKernel("matmul16x32");
    clSetKernelArg(lm_kernel, 0, sizeof(cl_mem), &lm_head->buffer);
    clSetKernelArg(lm_kernel, 1, sizeof(cl_mem), &final_normed->buffer);
    clSetKernelArg(lm_kernel, 2, sizeof(cl_mem), &logits_out->buffer);
    int lm_rows = VOCAB_SIZE;
    int lm_cols = HIDDEN_DIM;
    clSetKernelArg(lm_kernel, 3, sizeof(int), &lm_rows);
    clSetKernelArg(lm_kernel, 4, sizeof(int), &lm_cols);
    
    // NEW: Local memory for vector caching and reduction
    size_t local_vec_size = HIDDEN_DIM * sizeof(float);  // 4096 * 4 = 16KB
    size_t reduce_buf_size = 256 * sizeof(float);        // 256 * 4 = 1KB
    clSetKernelArg(lm_kernel, 5, local_vec_size, nullptr);
    clSetKernelArg(lm_kernel, 6, reduce_buf_size, nullptr);
    
    // NEW: Launch with one workgroup (256 threads) per row
    size_t lm_global_size = (size_t)VOCAB_SIZE * 256;  // 32768 * 256
    size_t lm_local_size = 256;
    gpu->enqueueKernel("matmul16x32", lm_kernel, 1, nullptr, &lm_global_size, &lm_local_size);
    
    // ===== ONLY eval() IN THE ENTIRE FORWARD PASS =====
    gpu->eval();
}
    
    void reset_caches() {
        k_cache->zero(gpu);
        v_cache->zero(gpu);
        gpu->eval();
    }
};

#endif