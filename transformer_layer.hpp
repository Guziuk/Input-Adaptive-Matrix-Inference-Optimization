// transformer_layer.hpp - FIXED VERSION - NO eval() calls inside
// UPDATED: All kernels now tracked via gpu->enqueueKernel()
#ifndef TRANSFORMER_LAYER_HPP
#define TRANSFORMER_LAYER_HPP

#include "gpu.hpp"
#include "attention.hpp"
#include "feedforward.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <cmath>

class TransformerLayer {
private:
    Gpu* gpu;
    std::unique_ptr<Attention> attention;
    std::unique_ptr<FeedForward> feedforward;
    
    BufferWrapper<uint16_t>* attn_norm_weight;
    BufferWrapper<uint16_t>* ffn_norm_weight;
    
    static constexpr int HIDDEN_DIM = 4096;
    
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
    TransformerLayer(Gpu* g, const std::string& weights_dir, int layer_idx)
        : gpu(g) {
        
        attention = std::make_unique<Attention>(gpu, weights_dir, layer_idx);
        feedforward = std::make_unique<FeedForward>(gpu, weights_dir, layer_idx);
        
        auto attn_norm_data = load_matrix_fp16(weights_dir + "/layer" + std::to_string(layer_idx) + "_input_layernorm.npy");
        attn_norm_weight = new BufferWrapper<uint16_t>(gpu, HIDDEN_DIM);
        attn_norm_weight->setData(gpu, attn_norm_data);
        
        auto ffn_norm_data = load_matrix_fp16(weights_dir + "/layer" + std::to_string(layer_idx) + "_post_attention_layernorm.npy");
        ffn_norm_weight = new BufferWrapper<uint16_t>(gpu, HIDDEN_DIM);
        ffn_norm_weight->setData(gpu, ffn_norm_data);
    }
    
    ~TransformerLayer() {
        delete attn_norm_weight;
        delete ffn_norm_weight;
    }
    
    void forward(BufferWrapper<float>* input,
                 int layer_idx,
                 int pos,
                 float effort,
                 BufferWrapper<float>* k_cache,
                 BufferWrapper<float>* v_cache,
                 BufferWrapper<float>* h_norm,
                 BufferWrapper<float>* temp_xq,
                 BufferWrapper<float>* temp_xk,
                 BufferWrapper<float>* temp_xv,
                 BufferWrapper<float>* temp_xk_repeated,
                 BufferWrapper<float>* temp_xq_heads,
                 BufferWrapper<float>* temp_xk_heads,
                 BufferWrapper<float>* temp_scores,
                 BufferWrapper<float>* temp_attn_output,
                 BufferWrapper<float>* attn_out,
                 BufferWrapper<float>* fxn,
                 BufferWrapper<float>* ffn_out,
                 BufferWrapper<float>* output) {
        
        // ===== ATTENTION NORM =====
        rms_norm(input, attn_norm_weight, h_norm);
        // NO eval()
        
        // ===== ATTENTION =====
        attention->forward(h_norm, layer_idx, pos, effort, 
                          k_cache, v_cache,
                          temp_xq, temp_xk, temp_xv, temp_xk_repeated,
                          temp_xq_heads, temp_xk_heads, temp_scores, temp_attn_output,
                          attn_out);
        // NO eval()
        
        // ===== RESIDUAL CONNECTION =====
        add_residual(input, attn_out, output);
        // NO eval()
        
        // ===== FFN NORM =====
        rms_norm(output, ffn_norm_weight, fxn);
        // NO eval()
        
        // ===== FEED FORWARD =====
        feedforward->forward(fxn, ffn_out, effort);
        // NO eval()
        
        // ===== FINAL RESIDUAL CONNECTION =====
        add_residual(output, ffn_out, output);
        // NO eval() - let model.forward() sync at the very end
    }

private:
    void rms_norm(BufferWrapper<float>* input, BufferWrapper<uint16_t>* weight, BufferWrapper<float>* output) {
        cl_kernel kernel = gpu->getKernel("rmsNorm32fast");
        
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->buffer);
        cl_uint count = HIDDEN_DIM;
        err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &count);
        err |= clSetKernelArg(kernel, 3, 256 * sizeof(float), nullptr);
        
        size_t global_size = 256;
        size_t local_size = 256;
        
        // TRACKED
        gpu->enqueueKernel("rmsNorm32fast", kernel, 1, nullptr, &global_size, &local_size);
        // NO eval()
        
        // Multiply by norm weights
        cl_kernel mul_kernel = gpu->getKernel("mul16x32");
        clSetKernelArg(mul_kernel, 0, sizeof(cl_mem), &weight->buffer);
        clSetKernelArg(mul_kernel, 1, sizeof(cl_mem), &output->buffer);
        clSetKernelArg(mul_kernel, 2, sizeof(cl_mem), &output->buffer);
        
        // TRACKED
        gpu->enqueueKernel1D("mul16x32", mul_kernel, HIDDEN_DIM);
        // NO eval()
    }
    
    void add_residual(BufferWrapper<float>* residual, BufferWrapper<float>* input, BufferWrapper<float>* output) {
        cl_kernel kernel = gpu->getKernel("add32");
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &residual->buffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->buffer);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->buffer);
        
        // TRACKED
        gpu->enqueueKernel1D("add32", kernel, HIDDEN_DIM);
        // NO eval()
    }
};

#endif