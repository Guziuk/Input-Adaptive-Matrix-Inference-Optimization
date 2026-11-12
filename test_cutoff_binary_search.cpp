// test_cutoff_binary_search.cpp - Test harness for findCutoffBinarySearch kernel
// Features:
// - Load real weights from weights directory
// - Test with different effort levels
// - Visualize importance distribution
// - Compare results with histogram method
// - Export data for Python plotting

#include "gpu.hpp"
#include "bucket_mul.hpp"  // CRITICAL: Needed for BufferWrapper
#include "binary_loader.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <cstring>

// Configuration
struct TestConfig {
    std::string weights_dir = "../weights";
    int layer = 0;
    std::string weight_type = "q_proj";  // q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    float effort = 0.5f;  // 0.0 to 1.0
    bool save_csv = false;
    bool verbose = false;
    std::string input_type = "random";  // "random", "zeros", "ones", "gaussian"
};

// Statistics structure
struct CutoffStats {
    float cutoff_value;
    int num_above_cutoff;
    int num_below_cutoff;
    float pct_above;
    float pct_below;
    double execution_time_ms;
    
    // Distribution stats
    float min_importance;
    float max_importance;
    float mean_importance;
    float median_importance;
    float std_importance;
};

class CutoffTester {
private:
    Gpu* gpu;
    BinaryWeightLoader* loader;
    TestConfig config;
    
    static constexpr int PROBES_COUNT = 4096;
    static constexpr float CUTOFF_SCALE = 100000.0f;
    
    BufferWrapper<float>* input_vec;
    BufferWrapper<uint16_t>* probes;
    BufferWrapper<uint16_t>* buckets;  // Actual bucket weights for zero analysis
    BufferWrapper<int>* expert_no;
    BufferWrapper<float>* cutoff_result;
    
    std::vector<float> importance_values;
    
public:
    CutoffTester(Gpu* g, const TestConfig& cfg) 
        : gpu(g), loader(nullptr), config(cfg),
          input_vec(nullptr), probes(nullptr), buckets(nullptr), expert_no(nullptr), cutoff_result(nullptr) {
        
        loader = new BinaryWeightLoader(gpu, config.weights_dir);
        
        std::cout << "=== Cutoff Binary Search Test Harness ===\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Layer: " << config.layer << "\n";
        std::cout << "  Weight type: " << config.weight_type << "\n";
        std::cout << "  Effort: " << config.effort << "\n";
        std::cout << "  Input type: " << config.input_type << "\n\n";
    }
    
    ~CutoffTester() {
        delete input_vec;
        delete probes;
        delete buckets;
        delete expert_no;
        delete cutoff_result;
        delete loader;
    }
    
    void loadWeights() {
        std::cout << "Loading weights...\n";
        
        // Load bucketized weight
        std::ostringstream oss;
        oss << "layer" << config.layer << "_" << config.weight_type << "_bucketmul_fp16";
        auto weight = loader->load(oss.str());
        
        std::cout << "  Loaded: " << oss.str() << "\n";
        std::cout << "  Input dim: " << weight.meta.in_dim << "\n";
        std::cout << "  Output dim: " << weight.meta.out_dim << "\n";
        std::cout << "  Probes shape: (" << weight.meta.probes_shape.dim0 
                  << ", " << weight.meta.probes_shape.dim1 << ")\n\n";
        
        // Create input vector
        input_vec = new BufferWrapper<float>(gpu, PROBES_COUNT);
        std::vector<float> input_data = generateInput();
        input_vec->setData(gpu, input_data);
        
        // Upload probes
        probes = new BufferWrapper<uint16_t>(gpu, {weight.meta.probes_shape.dim0, 
                                                     weight.meta.probes_shape.dim1});
        probes->setData(gpu, weight.probes_host);
        
        // Upload buckets for zero analysis
        buckets = new BufferWrapper<uint16_t>(gpu, {weight.meta.buckets_shape.dim0 * weight.meta.buckets_shape.dim1,
                                                     weight.meta.buckets_shape.dim2});
        buckets->setData(gpu, weight.buckets_host);
        
        // Expert number (0 for Mistral)
        expert_no = new BufferWrapper<int>(gpu, 1);
        std::vector<int> exp_no = {0};
        expert_no->setData(gpu, exp_no);
        
        // Output buffer
        cutoff_result = new BufferWrapper<float>(gpu, 1);
        
        std::cout << "[OK] Weights loaded successfully\n\n";
    }
    
    std::vector<float> generateInput() {
        std::vector<float> input(PROBES_COUNT);
        
        if (config.input_type == "zeros") {
            std::fill(input.begin(), input.end(), 0.0f);
        }
        else if (config.input_type == "ones") {
            std::fill(input.begin(), input.end(), 1.0f);
        }
        else if (config.input_type == "random") {
            for (int i = 0; i < PROBES_COUNT; i++) {
                input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
            }
        }
        else if (config.input_type == "gaussian") {
            // Box-Muller transform for gaussian
            for (int i = 0; i < PROBES_COUNT; i += 2) {
                float u1 = (float)rand() / RAND_MAX;
                float u2 = (float)rand() / RAND_MAX;
                float r = std::sqrt(-2.0f * std::log(u1));
                float theta = 2.0f * M_PI * u2;
                input[i] = r * std::cos(theta);
                if (i + 1 < PROBES_COUNT) {
                    input[i + 1] = r * std::sin(theta);
                }
            }
        }
        
        std::cout << "Generated input (" << config.input_type << "):\n";
        std::cout << "  Sample values: [";
        for (int i = 0; i < 5; i++) {
            std::cout << input[i];
            if (i < 4) std::cout << ", ";
        }
        std::cout << ", ...]\n";
        
        return input;
    }
    
    CutoffStats runTest() {
        std::cout << "Running findCutoffBinarySearch kernel...\n";
        
        int quantile = (int)((double)(PROBES_COUNT - 1) * (1.0 - config.effort));
        std::cout << "  Effort: " << config.effort << " -> Quantile: " << quantile << "\n";
        std::cout << "  Target elements above cutoff: " << (PROBES_COUNT - quantile) << "\n\n";
        
        // Run kernel
        cl_kernel kernel = gpu->getKernel("findCutoffBinarySearch");
        cl_int err;
        
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_vec->buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &probes->buffer);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &expert_no->buffer);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cutoff_result->buffer);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &quantile);
        err |= clSetKernelArg(kernel, 5, 1024 * sizeof(float), nullptr);
        
        size_t global_size = 256;
        size_t local_size = 256;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        err = gpu->enqueueKernel("findCutoffBinarySearch", kernel, 1, 
                                nullptr, &global_size, &local_size);
        gpu->eval();
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Kernel execution failed");
        }
        
        // Read result
        std::vector<float> cutoff_data = cutoff_result->getData(gpu);
        float cutoff = cutoff_data[0];
        
        std::cout << "[OK] Kernel completed in " << std::fixed << std::setprecision(3) 
                  << time_ms << " ms\n";
        std::cout << "  Cutoff value: " << cutoff << "\n\n";
        
        // Compute importance values on CPU for analysis
        std::cout << "Computing importance values for analysis...\n";
        computeImportanceValues();
        
        // Analyze results
        CutoffStats stats = analyzeResults(cutoff, time_ms);
        
        return stats;
    }
    
    void computeImportanceValues() {
        // Read input and probes from GPU
        std::vector<float> input = input_vec->getData(gpu);
        std::vector<uint16_t> probes_data = probes->getData(gpu);
        
        importance_values.resize(PROBES_COUNT);
        
        for (int i = 0; i < PROBES_COUNT; i++) {
            uint16_t probe_bits = probes_data[i];  // Expert 0 offset
            float probe_val = fp16_to_float(probe_bits);
            importance_values[i] = CUTOFF_SCALE * std::abs(input[i] * probe_val);
        }
        
        std::cout << "  Computed " << PROBES_COUNT << " importance values\n";
    }
    
    CutoffStats analyzeResults(float cutoff, double time_ms) {
        CutoffStats stats;
        stats.cutoff_value = cutoff;
        stats.execution_time_ms = time_ms;
        
        // Count above/below
        int above = 0, below = 0;
        for (float imp : importance_values) {
            if (imp > cutoff) above++;
            else below++;
        }
        
        stats.num_above_cutoff = above;
        stats.num_below_cutoff = below;
        stats.pct_above = 100.0f * above / PROBES_COUNT;
        stats.pct_below = 100.0f * below / PROBES_COUNT;
        
        // Distribution statistics
        std::vector<float> sorted = importance_values;
        std::sort(sorted.begin(), sorted.end());
        
        stats.min_importance = sorted.front();
        stats.max_importance = sorted.back();
        stats.median_importance = sorted[PROBES_COUNT / 2];
        
        // Mean
        float sum = 0;
        for (float v : importance_values) sum += v;
        stats.mean_importance = sum / PROBES_COUNT;
        
        // Std dev
        float var_sum = 0;
        for (float v : importance_values) {
            float diff = v - stats.mean_importance;
            var_sum += diff * diff;
        }
        stats.std_importance = std::sqrt(var_sum / PROBES_COUNT);
        
        return stats;
    }
    
    void printStats(const CutoffStats& stats) {
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "                   CUTOFF ANALYSIS RESULTS                      \n";
        std::cout << "================================================================\n";
        
        std::cout << std::fixed << std::setprecision(2);
        
        std::cout << "Cutoff Value:           " << std::setw(15) << stats.cutoff_value << "\n";
        std::cout << "Execution Time:         " << std::setw(10) << stats.execution_time_ms << " ms\n";
        std::cout << "----------------------------------------------------------------\n";
        
        std::cout << "Elements Above Cutoff:  " << std::setw(10) << stats.num_above_cutoff 
                  << " (" << std::setw(5) << stats.pct_above << "%)\n";
        std::cout << "Elements Below Cutoff:  " << std::setw(10) << stats.num_below_cutoff
                  << " (" << std::setw(5) << stats.pct_below << "%)\n";
        std::cout << "----------------------------------------------------------------\n";
        
        std::cout << "Min Importance:         " << std::setw(15) << stats.min_importance << "\n";
        std::cout << "Max Importance:         " << std::setw(15) << stats.max_importance << "\n";
        std::cout << "Mean Importance:        " << std::setw(15) << stats.mean_importance << "\n";
        std::cout << "Median Importance:      " << std::setw(15) << stats.median_importance << "\n";
        std::cout << "Std Dev:                " << std::setw(15) << stats.std_importance << "\n";
        
        std::cout << "================================================================\n";
    }
    
    void visualizeDistribution(const CutoffStats& stats) {
        std::cout << "\n=== RAW IMPORTANCE VALUES (sorted descending) ===\n\n";
        
        // Sort values descending to see distribution
        std::vector<float> sorted_vals = importance_values;
        std::sort(sorted_vals.begin(), sorted_vals.end(), std::greater<float>());
        
        // Statistics
        float min_val = sorted_vals.back();
        float max_val = sorted_vals.front();
        int num_zeros = std::count_if(sorted_vals.begin(), sorted_vals.end(), 
                                       [](float v) { return v == 0.0f; });
        int num_near_zero = std::count_if(sorted_vals.begin(), sorted_vals.end(),
                                          [](float v) { return v < 1.0f; });
        
        std::cout << "Total values: " << sorted_vals.size() << "\n";
        std::cout << "Max value: " << std::fixed << std::setprecision(2) << max_val << "\n";
        std::cout << "Min value: " << std::fixed << std::setprecision(2) << min_val << "\n";
        std::cout << "Zeros: " << num_zeros << " (" << std::fixed << std::setprecision(1) 
                  << (num_zeros * 100.0f / PROBES_COUNT) << "%)\n";
        std::cout << "Near-zero (<1.0): " << num_near_zero << " (" 
                  << (num_near_zero * 100.0f / PROBES_COUNT) << "%)\n";
        std::cout << "Cutoff value: " << std::fixed << std::setprecision(2) 
                  << stats.cutoff_value << "\n";
        
        // Find cutoff position in sorted array
        int cutoff_pos = 0;
        for (size_t i = 0; i < sorted_vals.size(); i++) {
            if (sorted_vals[i] <= stats.cutoff_value) {
                cutoff_pos = i;
                break;
            }
        }
        std::cout << "Cutoff position: " << cutoff_pos << " / " << PROBES_COUNT 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (cutoff_pos * 100.0f / PROBES_COUNT) << "% selected)\n\n";
        
        // Show top values (potential overhead creators)
        std::cout << "TOP 50 VALUES (highest importance - always selected):\n";
        for (int i = 0; i < std::min(50, (int)sorted_vals.size()); i++) {
            std::cout << std::setw(4) << i << ": " << std::setw(12) << std::fixed 
                      << std::setprecision(2) << sorted_vals[i];
            if (i % 4 == 3) std::cout << "\n";
        }
        if (50 % 4 != 0) std::cout << "\n";
        
        // Show values around cutoff
        std::cout << "\nVALUES AROUND CUTOFF (position " << cutoff_pos << "):\n";
        int start = std::max(0, cutoff_pos - 10);
        int end = std::min((int)sorted_vals.size(), cutoff_pos + 10);
        for (int i = start; i < end; i++) {
            std::cout << std::setw(4) << i << ": " << std::setw(12) << std::fixed 
                      << std::setprecision(2) << sorted_vals[i];
            if (i == cutoff_pos) std::cout << " <-- CUTOFF";
            std::cout << "\n";
        }
        
        // Show bottom values (zeros/near-zeros that create overhead)
        std::cout << "\nBOTTOM 50 VALUES (lowest importance - overhead if selected):\n";
        int bottom_start = std::max(0, (int)sorted_vals.size() - 50);
        int count = 0;
        for (int i = bottom_start; i < sorted_vals.size(); i++) {
            std::cout << std::setw(4) << i << ": " << std::setw(12) << std::fixed 
                      << std::setprecision(2) << sorted_vals[i];
            count++;
            if (count % 4 == 0) std::cout << "\n";
        }
        if (count % 4 != 0) std::cout << "\n";
        
        // Show distribution summary
        std::cout << "\n=== DISTRIBUTION SUMMARY ===\n";
        std::cout << "Values > 1000: " << std::count_if(sorted_vals.begin(), sorted_vals.end(),
                                                         [](float v) { return v > 1000.0f; }) << "\n";
        std::cout << "Values > 100: " << std::count_if(sorted_vals.begin(), sorted_vals.end(),
                                                        [](float v) { return v > 100.0f; }) << "\n";
        std::cout << "Values > 10: " << std::count_if(sorted_vals.begin(), sorted_vals.end(),
                                                       [](float v) { return v > 10.0f; }) << "\n";
        std::cout << "Values > 1: " << std::count_if(sorted_vals.begin(), sorted_vals.end(),
                                                      [](float v) { return v > 1.0f; }) << "\n";
        std::cout << "Values 0-1: " << num_near_zero << "\n";
        std::cout << "Exact zeros: " << num_zeros << "\n";
        
        std::cout << "\nIMPLICATIONS FOR BUCKETMUL:\n";
        std::cout << "  - High values at top are efficiently processed\n";
        std::cout << "  - Zero/near-zero values at bottom create overhead if selected\n";
        std::cout << "  - At effort=" << config.effort << ", selecting top " 
                  << std::fixed << std::setprecision(1) << (cutoff_pos * 100.0f / PROBES_COUNT) 
                  << "% avoids most overhead\n";
        
        // CRITICAL ANALYSIS: Check actual zeros in selected buckets
        std::cout << "\n=== ACTUAL ZERO OVERHEAD IN SELECTED BUCKETS ===\n";
        analyzeActualZeros(sorted_vals, cutoff_pos, stats);
    }
    
    void analyzeActualZeros(const std::vector<float>& sorted_importance, int cutoff_pos, const CutoffStats& stats) {
        // For each selected input dimension, check how many actual bucket values are zero
        std::vector<int> selected_indices;
        
        // Find which input dimensions are selected
        for (int i = 0; i < PROBES_COUNT; i++) {
            if (importance_values[i] >= stats.cutoff_value) {
                selected_indices.push_back(i);
            }
        }
        
        std::cout << "Analyzing " << selected_indices.size() << " selected input dimensions...\n\n";
        
        // Read actual bucket data from GPU
        std::vector<uint16_t> buckets_data = buckets->getData(gpu);
        
        // buckets shape: (1, in_dim*16, out_dim/16)
        // For each selected input_dim, we have 16 bucket rows
        size_t num_bucket_cols = buckets->cols;  // out_dim/16
        
        int total_values = 0;
        int total_zeros = 0;
        int total_near_zeros_001 = 0;  // < 0.01
        int total_near_zeros_01 = 0;   // < 0.1
        int total_near_zeros_1 = 0;    // < 1.0
        
        // Sample analysis: first 100 selected dims
        int samples_to_check = std::min(100, (int)selected_indices.size());
        
        for (int s = 0; s < samples_to_check; s++) {
            int input_dim = selected_indices[s];
            
            // This input_dim corresponds to 16 bucket rows (bsize=16)
            int bucket_row_start = input_dim * 16;
            
            int dim_zeros = 0;
            int dim_values = 0;
            
            for (int b = 0; b < 16; b++) {
                int bucket_row = bucket_row_start + b;
                
                for (size_t col = 0; col < num_bucket_cols; col++) {
                    size_t idx = bucket_row * num_bucket_cols + col;
                    if (idx < buckets_data.size()) {
                        float val = std::abs(extract_value(buckets_data[idx]));
                        dim_values++;
                        if (val == 0.0f) {
                            dim_zeros++;
                        }
                        if (val < 0.01f) {
                            total_near_zeros_001++;
                        }
                        if (val < 0.1f) {
                            total_near_zeros_01++;
                        }
                        if (val < 1.0f) {
                            total_near_zeros_1++;
                        }
                    }
                }
            }
            
            total_values += dim_values;
            total_zeros += dim_zeros;
        }
        
        float zero_pct = (total_values > 0) ? (total_zeros * 100.0f / total_values) : 0.0f;
        float near_zero_001_pct = (total_values > 0) ? (total_near_zeros_001 * 100.0f / total_values) : 0.0f;
        float near_zero_01_pct = (total_values > 0) ? (total_near_zeros_01 * 100.0f / total_values) : 0.0f;
        float near_zero_1_pct = (total_values > 0) ? (total_near_zeros_1 * 100.0f / total_values) : 0.0f;
        
        std::cout << "Sampled " << samples_to_check << " selected input dimensions:\n";
        std::cout << "  Total weight values: " << total_values << "\n";
        std::cout << "  Exact zeros: " << total_zeros << " (" << std::fixed << std::setprecision(1) 
                  << zero_pct << "%)\n";
        std::cout << "  Near-zero (<0.01): " << total_near_zeros_001 << " (" 
                  << near_zero_001_pct << "%)\n";
        std::cout << "  Small (<0.1): " << total_near_zeros_01 << " (" 
                  << near_zero_01_pct << "%)\n";
        std::cout << "  Modest (<1.0): " << total_near_zeros_1 << " (" 
                  << near_zero_1_pct << "%)\n\n";
        
        std::cout << "CRITICAL INSIGHT:\n";
        std::cout << "  Even with HIGH importance, " << std::fixed << std::setprecision(1) 
                  << near_zero_001_pct << "% of bucket values are < 0.01!\n";
        std::cout << "  These tiny values contribute almost nothing but cost full GPU cycles.\n";
        std::cout << "  \n";
        std::cout << "  OVERHEAD BREAKDOWN:\n";
        std::cout << "    - Exact zeros:  " << zero_pct << "% (skippable)\n";
        std::cout << "    - Tiny (<0.01): " << near_zero_001_pct << "% (negligible contribution)\n";
        std::cout << "    - Small (<0.1): " << near_zero_01_pct << "% (low contribution)\n";
        std::cout << "    - Significant:  " << std::fixed << std::setprecision(1) 
                  << (100.0f - near_zero_01_pct) << "% (actually useful work)\n";
        std::cout << "  \n";
        std::cout << "  SOLUTION: Sparse bucketMul could skip values < 0.1 for ~" 
                  << std::fixed << std::setprecision(0) << near_zero_01_pct << "% speedup!\n";
    }
    
    void saveToCSV(const CutoffStats& stats, const std::string& filename) {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to open " << filename << " for writing\n";
            return;
        }
        
        file << "# Cutoff Binary Search Test Results\n";
        file << "# Layer: " << config.layer << "\n";
        file << "# Weight: " << config.weight_type << "\n";
        file << "# Effort: " << config.effort << "\n";
        file << "# Cutoff: " << stats.cutoff_value << "\n";
        file << "# Above: " << stats.num_above_cutoff << "\n";
        file << "# Below: " << stats.num_below_cutoff << "\n";
        file << "#\n";
        file << "importance_value\n";
        
        for (float val : importance_values) {
            file << val << "\n";
        }
        
        file.close();
        std::cout << "\n[OK] Saved importance values to " << filename << "\n";
        std::cout << "  You can plot this with: python plot_cutoff.py " << filename << "\n";
    }
    
    
private:
    float extract_value(uint16_t packed_fp16) {
        // Clear lower 4 bits (position info) to get just the value
        uint16_t value_bits = packed_fp16 & 0xFFF0;
        return fp16_to_float(value_bits);
    }
    
    float fp16_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = (h & 0x03FF);
        
        if (exponent == 0) {
            if (mantissa == 0) {
                uint32_t result = sign;
                return *reinterpret_cast<float*>(&result);
            } else {
                float value = std::ldexp((float)mantissa, -24);
                return (sign != 0) ? -value : value;
            }
        } else if (exponent == 31) {
            uint32_t result = sign | 0x7F800000 | (mantissa << 13);
            return *reinterpret_cast<float*>(&result);
        } else {
            exponent = exponent - 15 + 127;
            uint32_t result = sign | (exponent << 23) | (mantissa << 13);
            return *reinterpret_cast<float*>(&result);
        }
    }
};

int main(int argc, char** argv) {
    TestConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--layer" && i + 1 < argc) {
            config.layer = std::stoi(argv[++i]);
        }
        else if (arg == "--weight" && i + 1 < argc) {
            config.weight_type = argv[++i];
        }
        else if (arg == "--effort" && i + 1 < argc) {
            config.effort = std::stof(argv[++i]);
        }
        else if (arg == "--input" && i + 1 < argc) {
            config.input_type = argv[++i];
        }
        else if (arg == "--save-csv") {
            config.save_csv = true;
        }
        else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --layer N         Layer number (default: 0)\n";
            std::cout << "  --weight TYPE     Weight type: q_proj, k_proj, v_proj, o_proj,\n";
            std::cout << "                    gate_proj, up_proj, down_proj (default: q_proj)\n";
            std::cout << "  --effort F        Effort level 0.0-1.0 (default: 0.5)\n";
            std::cout << "  --input TYPE      Input type: random, zeros, ones, gaussian (default: random)\n";
            std::cout << "  --save-csv        Save importance values to CSV\n";
            std::cout << "  --verbose, -v     Verbose output\n";
            std::cout << "  --help, -h        Show this help\n\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << " --layer 0 --weight q_proj --effort 0.8\n";
            std::cout << "  " << argv[0] << " --input gaussian --save-csv\n";
            return 0;
        }
    }
    
    try {
        Gpu* gpu = new Gpu();
        
        CutoffTester tester(gpu, config);
        
        // Load weights
        tester.loadWeights();
        
        // Run test
        auto stats = tester.runTest();
        
        // Print results
        tester.printStats(stats);
        
        // Visualize
        tester.visualizeDistribution(stats);
        
        // Save to CSV if requested
        if (config.save_csv) {
            std::string filename = "cutoff_test_layer" + std::to_string(config.layer) + 
                                  "_" + config.weight_type + ".csv";
            tester.saveToCSV(stats, filename);
        }
        
        delete gpu;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}