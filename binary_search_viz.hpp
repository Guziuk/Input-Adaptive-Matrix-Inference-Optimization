// Binary Search Convergence Visualization
// Shows how the binary search narrows down to the exact cutoff

#include "gpu.hpp"
#include "bucket_mul.hpp"
#include "binary_loader.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>

#define CUTOFF_SCALE 100000.0f
#define PROBES_COUNT 4096

// Convergence tracking structure
struct ConvergenceStep {
    int iteration;
    float minBound;
    float maxBound;
    float testPoint;
    int countAbove;
    int target;
    bool converged;
    float range_size;
};

class BinarySearchAnalyzer {
private:
    Gpu* gpu;
    std::vector<float> importance_values;
    std::vector<ConvergenceStep> convergence_history;
    float final_cutoff;
    int target_count;
    
public:
    BinarySearchAnalyzer(Gpu* g) : gpu(g), final_cutoff(0.0f), target_count(0) {}
    
    void setImportanceValues(const std::vector<float>& vals) {
        importance_values = vals;
    }
    
    void setTarget(int target) {
        target_count = target;
    }
    
    void trackConvergence(int iter, float min_b, float max_b, float test, int count, bool conv) {
        ConvergenceStep step;
        step.iteration = iter;
        step.minBound = min_b;
        step.maxBound = max_b;
        step.testPoint = test;
        step.countAbove = count;
        step.target = target_count;
        step.converged = conv;
        step.range_size = max_b - min_b;
        convergence_history.push_back(step);
    }
    
    void setFinalCutoff(float cutoff) {
        final_cutoff = cutoff;
    }
    
    // Print convergence trace
    void printConvergenceTrace() {
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "           BINARY SEARCH CONVERGENCE TRACE\n";
        std::cout << "================================================================\n";
        std::cout << "Target: Select exactly " << target_count << " elements\n\n";
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Iter | Test Point | Count Above | Target | Range Size | Status\n";
        std::cout << "-----+------------+-------------+--------+------------+-----------\n";
        
        for (const auto& step : convergence_history) {
            std::cout << std::setw(4) << step.iteration << " | ";
            std::cout << std::setw(10) << step.testPoint << " | ";
            std::cout << std::setw(11) << step.countAbove << " | ";
            std::cout << std::setw(6) << step.target << " | ";
            std::cout << std::setw(10) << step.range_size << " | ";
            
            if (step.converged) {
                std::cout << "CONVERGED";
            } else if (step.countAbove < step.target) {
                std::cout << "Too few  (lower bound)";
            } else if (step.countAbove > step.target) {
                std::cout << "Too many (raise bound)";
            } else {
                std::cout << "EXACT!";
            }
            std::cout << "\n";
        }
        
        std::cout << "-----+------------+-------------+--------+------------+-----------\n";
        std::cout << "Final cutoff: " << final_cutoff << "\n";
        std::cout << "Converged in " << convergence_history.size() << " iterations\n";
        std::cout << "================================================================\n\n";
    }
    
    // Visualize the cutoff point in sorted values
    void visualizeCutoffInSortedData() {
        if (importance_values.empty()) return;
        
        // Sort values
        std::vector<float> sorted = importance_values;
        std::sort(sorted.begin(), sorted.end());
        
        // Find where cutoff falls
        int cutoff_idx = 0;
        for (size_t i = 0; i < sorted.size(); i++) {
            if (sorted[i] > final_cutoff) {
                cutoff_idx = i;
                break;
            }
        }
        
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "          CUTOFF POSITION IN SORTED DATA\n";
        std::cout << "================================================================\n";
        std::cout << std::fixed << std::setprecision(2);
        
        // Show values around cutoff
        int window = 10;
        int start = std::max(0, cutoff_idx - window);
        int end = std::min((int)sorted.size(), cutoff_idx + window);
        
        std::cout << "Showing " << window << " values before and after cutoff:\n\n";
        std::cout << "Index | Value      | Selected?\n";
        std::cout << "------+------------+-----------\n";
        
        for (int i = start; i < end; i++) {
            std::cout << std::setw(5) << i << " | ";
            std::cout << std::setw(10) << sorted[i] << " | ";
            
            if (sorted[i] > final_cutoff) {
                std::cout << "YES  <-- SELECTED";
            } else {
                std::cout << "no";
            }
            
            if (i == cutoff_idx - 1) {
                std::cout << "  <-- CUTOFF = " << final_cutoff;
            }
            
            std::cout << "\n";
        }
        
        std::cout << "------+------------+-----------\n";
        std::cout << "Total above cutoff: " << (sorted.size() - cutoff_idx) << "\n";
        std::cout << "Total below cutoff: " << cutoff_idx << "\n";
        std::cout << "================================================================\n\n";
    }
    
    // Show distribution around cutoff
    void showDistributionAroundCutoff() {
        if (importance_values.empty()) return;
        
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "        DISTRIBUTION AROUND CUTOFF\n";
        std::cout << "================================================================\n";
        
        // Create ranges around cutoff
        float range = final_cutoff * 0.1f;  // ±10% of cutoff
        int num_buckets = 20;
        float bucket_width = (2 * range) / num_buckets;
        float start = final_cutoff - range;
        
        std::vector<int> buckets(num_buckets, 0);
        
        for (float val : importance_values) {
            if (val >= start && val < final_cutoff + range) {
                int bucket = (val - start) / bucket_width;
                if (bucket >= 0 && bucket < num_buckets) {
                    buckets[bucket]++;
                }
            }
        }
        
        // Find max for scaling
        int max_count = *std::max_element(buckets.begin(), buckets.end());
        int bar_width = 60;
        
        std::cout << std::fixed << std::setprecision(0);
        for (int i = 0; i < num_buckets; i++) {
            float bucket_start = start + i * bucket_width;
            float bucket_end = bucket_start + bucket_width;
            
            std::cout << std::setw(8) << bucket_start << " - " << std::setw(8) << bucket_end << " | ";
            
            // Draw bar
            int bar_len = (buckets[i] * bar_width) / std::max(1, max_count);
            for (int j = 0; j < bar_len; j++) {
                std::cout << "=";
            }
            std::cout << " " << buckets[i];
            
            // Mark cutoff
            if (bucket_start <= final_cutoff && final_cutoff < bucket_end) {
                std::cout << " <-- CUTOFF";
            }
            
            std::cout << "\n";
        }
        
        std::cout << "================================================================\n\n";
    }
    
    // Show percentile analysis
    void showPercentileAnalysis() {
        if (importance_values.empty()) return;
        
        std::vector<float> sorted = importance_values;
        std::sort(sorted.begin(), sorted.end());
        
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "               PERCENTILE ANALYSIS\n";
        std::cout << "================================================================\n";
        std::cout << std::fixed << std::setprecision(2);
        
        // Calculate percentiles
        std::vector<int> percentiles = {0, 10, 25, 50, 75, 90, 95, 99, 100};
        
        std::cout << "Percentile | Value      | Above Cutoff?\n";
        std::cout << "-----------+------------+--------------\n";
        
        for (int p : percentiles) {
            int idx = (p * (sorted.size() - 1)) / 100;
            float val = sorted[idx];
            
            std::cout << "   " << std::setw(3) << p << "%    | ";
            std::cout << std::setw(10) << val << " | ";
            std::cout << (val > final_cutoff ? "YES" : "no ");
            
            if (std::abs(val - final_cutoff) < 0.01f * final_cutoff) {
                std::cout << " <-- NEAR CUTOFF";
            }
            
            std::cout << "\n";
        }
        
        std::cout << "-----------+------------+--------------\n";
        std::cout << "Cutoff:      " << std::setw(10) << final_cutoff << "\n";
        
        // Find what percentile the cutoff represents
        int count_below = 0;
        for (float val : sorted) {
            if (val <= final_cutoff) count_below++;
            else break;
        }
        float cutoff_percentile = (100.0f * count_below) / sorted.size();
        
        std::cout << "Cutoff percentile: " << std::setw(5) << std::setprecision(1) 
                  << cutoff_percentile << "%\n";
        std::cout << "================================================================\n\n";
    }
    
    // Cumulative distribution function view
    void showCDF() {
        if (importance_values.empty()) return;
        
        std::vector<float> sorted = importance_values;
        std::sort(sorted.begin(), sorted.end());
        
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "      CUMULATIVE DISTRIBUTION FUNCTION (CDF)\n";
        std::cout << "================================================================\n";
        
        // Sample 20 points
        int num_points = 20;
        
        std::cout << "Value      | Cumulative % | Selected?\n";
        std::cout << "-----------+--------------+-----------\n";
        std::cout << std::fixed << std::setprecision(2);
        
        for (int i = 0; i < num_points; i++) {
            int idx = (i * (sorted.size() - 1)) / (num_points - 1);
            float val = sorted[idx];
            float cumulative_pct = (100.0f * (idx + 1)) / sorted.size();
            
            std::cout << std::setw(10) << val << " | ";
            std::cout << std::setw(11) << cumulative_pct << "% | ";
            std::cout << (val > final_cutoff ? "YES" : "no ");
            
            // Mark cutoff
            if (i > 0) {
                int prev_idx = ((i-1) * (sorted.size() - 1)) / (num_points - 1);
                float prev_val = sorted[prev_idx];
                
                if (prev_val <= final_cutoff && val > final_cutoff) {
                    std::cout << " <-- CUTOFF HERE";
                }
            }
            
            std::cout << "\n";
        }
        
        std::cout << "-----------+--------------+-----------\n";
        std::cout << "Cutoff: " << final_cutoff << "\n";
        std::cout << "================================================================\n\n";
    }
    
    // Export convergence data for plotting
    void exportConvergenceCSV(const std::string& filename) {
        std::ofstream out(filename);
        if (!out) {
            std::cerr << "Failed to open " << filename << "\n";
            return;
        }
        
        out << "iteration,min_bound,max_bound,test_point,count_above,target,range_size,converged\n";
        
        for (const auto& step : convergence_history) {
            out << step.iteration << ","
                << step.minBound << ","
                << step.maxBound << ","
                << step.testPoint << ","
                << step.countAbove << ","
                << step.target << ","
                << step.range_size << ","
                << (step.converged ? "1" : "0") << "\n";
        }
        
        out.close();
        std::cout << "Convergence data exported to: " << filename << "\n";
    }
    
    // Print comprehensive analysis
    void printFullAnalysis() {
        printConvergenceTrace();
        visualizeCutoffInSortedData();
        showDistributionAroundCutoff();
        showPercentileAnalysis();
        showCDF();
    }
};