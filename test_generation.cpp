// test_generation.cpp - With comprehensive timing metrics
#include "model.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>

// ============================================================================
// Timing Utilities
// ============================================================================

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running = false;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }
    
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }
    
    // Returns elapsed time in milliseconds
    double elapsed_ms() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }
    
    // Returns elapsed time in seconds
    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }
};

struct InferenceMetrics {
    // Timing
    double model_load_ms = 0;
    double prefill_ms = 0;
    double generation_ms = 0;
    double total_ms = 0;
    
    // Token counts
    int prompt_tokens = 0;
    int generated_tokens = 0;
    
    // Per-token times (for generation)
    std::vector<double> token_times_ms;
    
    // Computed metrics
    double prefill_tokens_per_sec() const {
        return (prefill_ms > 0) ? (prompt_tokens * 1000.0 / prefill_ms) : 0;
    }
    
    double generation_tokens_per_sec() const {
        return (generation_ms > 0) ? (generated_tokens * 1000.0 / generation_ms) : 0;
    }
    
    double avg_token_latency_ms() const {
        return (generated_tokens > 0) ? (generation_ms / generated_tokens) : 0;
    }
    
    double time_to_first_token_ms() const {
        return prefill_ms + (token_times_ms.empty() ? 0 : token_times_ms[0]);
    }
    
    void print() const {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║                   PERFORMANCE METRICS                    ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        
        std::cout << std::fixed << std::setprecision(2);
        
        // Timing breakdown
        std::cout << "║  Model Load Time:      " << std::setw(10) << model_load_ms << " ms" << std::setw(18) << "║\n";
        std::cout << "║  Prefill Time:         " << std::setw(10) << prefill_ms << " ms" << std::setw(18) << "║\n";
        std::cout << "║  Generation Time:      " << std::setw(10) << generation_ms << " ms" << std::setw(18) << "║\n";
        std::cout << "║  Total Inference:      " << std::setw(10) << (prefill_ms + generation_ms) << " ms" << std::setw(18) << "║\n";
        
        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        
        // Token counts
        std::cout << "║  Prompt Tokens:        " << std::setw(10) << prompt_tokens << std::setw(22) << "║\n";
        std::cout << "║  Generated Tokens:     " << std::setw(10) << generated_tokens << std::setw(22) << "║\n";
        
        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        
        // Throughput
        std::cout << "║  Prefill Speed:        " << std::setw(10) << prefill_tokens_per_sec() << " tok/s" << std::setw(16) << "║\n";
        std::cout << "║  Generation Speed:     " << std::setw(10) << generation_tokens_per_sec() << " tok/s" << std::setw(16) << "║\n";
        std::cout << "║  Avg Token Latency:    " << std::setw(10) << avg_token_latency_ms() << " ms" << std::setw(18) << "║\n";
        std::cout << "║  Time to First Token:  " << std::setw(10) << time_to_first_token_ms() << " ms" << std::setw(18) << "║\n";
        
        std::cout << "╚══════════════════════════════════════════════════════════╝\n";
        
        // Per-token breakdown (if verbose)
        if (!token_times_ms.empty()) {
            std::cout << "\n┌─ Per-Token Generation Times ─────────────────────────────┐\n";
            for (size_t i = 0; i < token_times_ms.size(); i++) {
                std::cout << "│  Token " << std::setw(2) << (i + 1) << ":  " 
                         << std::setw(8) << token_times_ms[i] << " ms";
                
                // Show relative speed
                double tok_per_sec = 1000.0 / token_times_ms[i];
                std::cout << "  (" << std::setw(6) << tok_per_sec << " tok/s)";
                std::cout << std::setw(10) << "│\n";
            }
            std::cout << "└──────────────────────────────────────────────────────────┘\n";
        }
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "=== Token Generation Test (32 Layers) ===\n\n";
    double effort = 1.0; 
    // Parse command line args
    int num_to_generate = 10;
    bool verbose = false;
    std::string prompt = "The capital of france is";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            num_to_generate = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-p" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "-e" && i + 1 < argc) {
            effort = std::stod(argv[++i]);
        }

         else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "  -n <num>     Number of tokens to generate (default: 10)\n";
            std::cout << "  -p <prompt>  Prompt text (default: \"The capital of france is\")\n";
            std::cout << "  -v           Verbose output\n";
            return 0;
        }
    }
    
    InferenceMetrics metrics;
    Timer timer;
    
    // ========================================================================
    // MODEL LOADING
    // ========================================================================
    timer.start();
    
    Gpu* gpu = new Gpu();
    
    std::cout << "Loading tokenizer...\n";
    Tokenizer tokenizer("../tokenizer.json");
    
    std::cout << "Loading model...\n";
    MistralModel model(gpu, "../weights");
    
    timer.stop();
    metrics.model_load_ms = timer.elapsed_ms();
    std::cout << "Model loaded in " << std::fixed << std::setprecision(2) 
              << metrics.model_load_ms << " ms\n\n";
    
    // ========================================================================
    // TOKENIZATION
    // ========================================================================
    auto prompt_tokens = tokenizer.encode(prompt);
    metrics.prompt_tokens = prompt_tokens.size();
    
    std::cout << "=== Configuration ===\n";
    std::cout << "Prompt: \"" << prompt << "\"\n";
    std::cout << "Prompt tokens: [";
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        std::cout << prompt_tokens[i];
        if (i < prompt_tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "] (" << prompt_tokens.size() << " tokens)\n";
    std::cout << "Will generate: " << num_to_generate << " new tokens\n\n";
    
    // ========================================================================
    // PHASE 1: PREFILL
    // ========================================================================
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║  PHASE 1: PREFILL                      ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    BufferWrapper<float> logits(gpu, 32768);
    
    timer.start();
     std::cout << effort<<"\n";
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        int token = prompt_tokens[i];
        if (verbose) {
            std::cout << "Position " << i << ": token [" << token << "] \"" 
                      << tokenizer.decode({token}) << "\"\n";
        }
        
        model.forward(token, i, effort, &logits);
    }
    
    timer.stop();
    metrics.prefill_ms = timer.elapsed_ms();
    
    std::cout << "✓ Prefill complete! " << prompt_tokens.size() << " tokens in " 
              << std::fixed << std::setprecision(2) << metrics.prefill_ms << " ms ("
              << metrics.prefill_tokens_per_sec() << " tok/s)\n\n";
    
    // ========================================================================
    // PHASE 2: GENERATION
    // ========================================================================
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║  PHASE 2: GENERATION                   ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::vector<int> generated_tokens;
    Timer gen_timer;
    Timer token_timer;
    
    // Read logits from last prefill position
    std::vector<float> logits_cpu(32768);
    gpu->readBuffer(logits.buffer, 32768 * sizeof(float), logits_cpu.data());
    
    // Find best token
    int next_token = std::max_element(logits_cpu.begin(), logits_cpu.end()) - logits_cpu.begin();
    
    gen_timer.start();
    
    for (int gen_step = 0; gen_step < num_to_generate; gen_step++) {
        token_timer.start();
        
        int pos = prompt_tokens.size() + gen_step;
        
        if (verbose) {
            std::cout << "┌─ Step " << (gen_step + 1) << "/" << num_to_generate 
                      << " (position " << pos << ") ─────────────\n";
            
            // Show top 5 predictions
            std::vector<std::pair<float, int>> top_logits;
            for (int j = 0; j < 32768; j++) {
                top_logits.push_back({logits_cpu[j], j});
            }
            std::partial_sort(top_logits.begin(), top_logits.begin() + 5, top_logits.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            
            std::cout << "│ Top 5 candidates:\n";
            for (int k = 0; k < 5; k++) {
                float logit = top_logits[k].first;
                int tok_id = top_logits[k].second;
                std::string tok_text = tokenizer.decode({tok_id});
                std::cout << "│   " << (k+1) << ". [" << tok_id << "] \"" << tok_text 
                         << "\" (logit: " << logit << ")\n";
            }
            
            std::cout << "│\n│ → Selected: [" << next_token << "] \"" 
                      << tokenizer.decode({next_token}) << "\"\n";
        }
        
        generated_tokens.push_back(next_token);
        
        // Forward pass with the selected token
        model.forward(next_token, pos, effort, &logits);
        
        // Read new logits for next iteration
        gpu->readBuffer(logits.buffer, 32768 * sizeof(float), logits_cpu.data());
        next_token = std::max_element(logits_cpu.begin(), logits_cpu.end()) - logits_cpu.begin();
        
        token_timer.stop();
        metrics.token_times_ms.push_back(token_timer.elapsed_ms());
        
        if (verbose) {
            std::cout << "│ Time: " << std::fixed << std::setprecision(2) 
                      << token_timer.elapsed_ms() << " ms\n";
            std::cout << "└────────────────────────────────────────\n";
        } else {
            // Simple progress output
            std::cout << "Generated token " << (gen_step + 1) << "/" << num_to_generate 
                      << ": \"" << tokenizer.decode({generated_tokens.back()}) << "\" ("
                      << std::fixed << std::setprecision(1) << token_timer.elapsed_ms() << " ms)\n";
        }
    }
    
    gen_timer.stop();
    metrics.generation_ms = gen_timer.elapsed_ms();
    metrics.generated_tokens = generated_tokens.size();
    
    // ========================================================================
    // RESULTS
    // ========================================================================
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  RESULTS                               ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::cout << "Prompt:     \"" << prompt << "\"\n";
    std::cout << "Generated:  \"" << tokenizer.decode(generated_tokens) << "\"\n\n";
    std::cout << "Full text:  \"" << prompt << tokenizer.decode(generated_tokens) << "\"\n";
    
    // Print metrics
    metrics.print();
    
    // Memory info
    std::cout << "\n";
    gpu->memTracker->print_status();
    
    delete gpu;
    return 0;
}