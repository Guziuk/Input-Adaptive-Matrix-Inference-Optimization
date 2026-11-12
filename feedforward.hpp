// feedforward.hpp - FIXED VERSION - NO eval() calls inside
// UPDATED: All kernels now tracked via gpu->enqueueKernel()
#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "gpu.hpp"
#include "bucket_mul.hpp"
#include "binary_loader.hpp"
#include <memory>
#include <string>
#include <vector>
#include <fstream>

class FeedForward {
private:
    Gpu* gpu;
    BinaryWeightLoader* loader;

    BufferWrapper<uint16_t>* gate_buckets;
    BufferWrapper<uint16_t>* gate_stats;
    BufferWrapper<uint16_t>* gate_probes;
    int gate_expert_size;
    int gate_bucket_cols;

    BufferWrapper<uint16_t>* up_buckets;
    BufferWrapper<uint16_t>* up_stats;
    BufferWrapper<uint16_t>* up_probes;
    int up_expert_size;
    int up_bucket_cols;

    BufferWrapper<uint16_t>* down_buckets;
    BufferWrapper<uint16_t>* down_stats;
    BufferWrapper<uint16_t>* down_probes;
    int down_expert_size;
    int down_bucket_cols;

    BufferWrapper<float>* gate_output;
    BufferWrapper<float>* up_output;
    BufferWrapper<float>* silu_output;

    size_t hidden_dim;
    size_t ffn_dim;
    
    int layer_idx_;

    void run_forward(BufferWrapper<float>* input,
                     BufferWrapper<float>* output,
                     float effort) {

        auto bm = BucketMul::shared(gpu);

        // Zero all outputs (non-blocking)
        gate_output->zero(gpu);
        up_output->zero(gpu);
        silu_output->zero(gpu);
        output->zero(gpu);
        // NO eval()
        
        // ===== GATE PROJECTION =====
        bm->calcDispatch(input, gate_stats, gate_probes, 0, static_cast<int>(hidden_dim), 
                        gate_bucket_cols, gate_expert_size, effort);
        bm->mul(gate_buckets, gate_output);
        // NO eval()
        
        // ===== UP PROJECTION =====
        bm->calcDispatch(input, up_stats, up_probes, 0, static_cast<int>(hidden_dim), 
                        up_bucket_cols, up_expert_size, effort);
        bm->mul(up_buckets, up_output);
        // NO eval()

        // ===== SILU ACTIVATION: silu(gate) * up =====
        cl_kernel silu = gpu->getKernel("silu32");
        clSetKernelArg(silu, 0, sizeof(cl_mem), &gate_output->buffer);
        clSetKernelArg(silu, 1, sizeof(cl_mem), &up_output->buffer);
        clSetKernelArg(silu, 2, sizeof(cl_mem), &silu_output->buffer);

        // TRACKED
        gpu->enqueueKernel1D("silu32", silu, ffn_dim);
        // NO eval()

        // ===== DOWN PROJECTION =====
        bm->calcDispatch(silu_output, down_stats, down_probes, 0, static_cast<int>(ffn_dim), 
                        down_bucket_cols, down_expert_size, effort);
        bm->mul(down_buckets, output);
        // NO eval() - let model.forward() sync at the very end
    }

public:
    FeedForward(Gpu* g,
                const std::string& weights_dir,
                int layer_idx,
                size_t hidden = 4096,
                size_t ffn = 14336)
        : gpu(g),
          loader(nullptr),
          gate_buckets(nullptr),
          gate_stats(nullptr),
          gate_probes(nullptr),
          up_buckets(nullptr),
          up_stats(nullptr),
          up_probes(nullptr),
          down_buckets(nullptr),
          down_stats(nullptr),
          down_probes(nullptr),
          gate_output(nullptr),
          up_output(nullptr),
          silu_output(nullptr),
          hidden_dim(hidden),
          ffn_dim(ffn),
          layer_idx_(layer_idx)
    {
        loader = new BinaryWeightLoader(gpu, weights_dir);
        std::string layer_prefix = "layer" + std::to_string(layer_idx);

        BucketizedWeight gate_w  = loader->load(layer_prefix + "_gate_proj_bucketmul_fp16");
        BucketizedWeight up_w    = loader->load(layer_prefix + "_up_proj_bucketmul_fp16");
        BucketizedWeight down_w  = loader->load(layer_prefix + "_down_proj_bucketmul_fp16");

        if (gate_w.meta.in_dim != 0)  hidden_dim = gate_w.meta.in_dim;
        if (gate_w.meta.out_dim != 0) ffn_dim    = gate_w.meta.out_dim;

        // Gate
        int gate_expert_count = static_cast<int>(gate_w.meta.stats_shape.dim0);
        gate_expert_size      = static_cast<int>(gate_w.meta.stats_shape.dim1);
        int gate_stats_channels = static_cast<int>(gate_w.meta.stats_shape.dim2);
        gate_bucket_cols = static_cast<int>(gate_w.meta.buckets_shape.dim2);
        size_t gate_bucket_rows = static_cast<size_t>(gate_expert_count) * static_cast<size_t>(gate_expert_size);

        gate_buckets = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{gate_bucket_rows, static_cast<size_t>(gate_bucket_cols)});
        gate_stats   = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{gate_bucket_rows, static_cast<size_t>(gate_stats_channels)});
        gate_probes  = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{gate_w.meta.probes_shape.dim0, gate_w.meta.probes_shape.dim1});

        gate_buckets->setData(gpu, gate_w.buckets_host);
        gate_stats->setData(gpu, gate_w.stats_host);
        gate_probes->setData(gpu, gate_w.probes_host);

        // Up
        int up_expert_count = static_cast<int>(up_w.meta.stats_shape.dim0);
        up_expert_size      = static_cast<int>(up_w.meta.stats_shape.dim1);
        int up_stats_channels = static_cast<int>(up_w.meta.stats_shape.dim2);
        up_bucket_cols = static_cast<int>(up_w.meta.buckets_shape.dim2);
        size_t up_bucket_rows = static_cast<size_t>(up_expert_count) * static_cast<size_t>(up_expert_size);

        up_buckets = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{up_bucket_rows, static_cast<size_t>(up_bucket_cols)});
        up_stats   = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{up_bucket_rows, static_cast<size_t>(up_stats_channels)});
        up_probes  = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{up_w.meta.probes_shape.dim0, up_w.meta.probes_shape.dim1});

        up_buckets->setData(gpu, up_w.buckets_host);
        up_stats->setData(gpu, up_w.stats_host);
        up_probes->setData(gpu, up_w.probes_host);

        // Down
        int down_expert_count = static_cast<int>(down_w.meta.stats_shape.dim0);
        down_expert_size      = static_cast<int>(down_w.meta.stats_shape.dim1);
        int down_stats_channels = static_cast<int>(down_w.meta.stats_shape.dim2);
        down_bucket_cols = static_cast<int>(down_w.meta.buckets_shape.dim2);
        size_t down_bucket_rows = static_cast<size_t>(down_expert_count) * static_cast<size_t>(down_expert_size);

        down_buckets = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{down_bucket_rows, static_cast<size_t>(down_bucket_cols)});
        down_stats   = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{down_bucket_rows, static_cast<size_t>(down_stats_channels)});
        down_probes  = new BufferWrapper<uint16_t>(gpu, std::vector<size_t>{down_w.meta.probes_shape.dim0, down_w.meta.probes_shape.dim1});

        down_buckets->setData(gpu, down_w.buckets_host);
        down_stats->setData(gpu, down_w.stats_host);
        down_probes->setData(gpu, down_w.probes_host);

        gate_output = new BufferWrapper<float>(gpu, ffn_dim);
        up_output   = new BufferWrapper<float>(gpu, ffn_dim);
        silu_output = new BufferWrapper<float>(gpu, ffn_dim);
    }

    ~FeedForward() {
        delete loader;
        delete gate_buckets; delete gate_stats; delete gate_probes;
        delete up_buckets; delete up_stats; delete up_probes;
        delete down_buckets; delete down_stats; delete down_probes;
        delete gate_output; delete up_output; delete silu_output;
    }

    void forward(BufferWrapper<float>* input, BufferWrapper<float>* output, float effort = 1.0f) {
        run_forward(input, output, effort);
    }
};

#endif