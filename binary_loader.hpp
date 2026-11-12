#ifndef BINARY_LOADER_HPP
#define BINARY_LOADER_HPP

#include "gpu.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <cmath>

// Simple structure to hold shape information
struct Shape3D {
    size_t dim0, dim1, dim2;
    
    size_t total() const { return dim0 * dim1 * dim2; }
    size_t bytes() const { return total() * sizeof(uint16_t); } // FP16 = 2 bytes
};

struct Shape2D {
    size_t dim0, dim1;
    
    size_t total() const { return dim0 * dim1; }
    size_t bytes() const { return total() * sizeof(uint16_t); }
};

// Metadata from the .txt file
struct WeightMetadata {
    Shape3D buckets_shape;
    Shape3D stats_shape;
    Shape2D probes_shape;
    size_t in_dim;
    size_t out_dim;
    size_t bsize;
};

// Complete bucketized weight set
struct BucketizedWeight {
    cl_mem buckets;
    cl_mem stats;
    cl_mem probes;
    
    WeightMetadata meta;
    
    // Host-side copies (for debugging/validation)
    std::vector<uint16_t> buckets_host;
    std::vector<uint16_t> stats_host;
    std::vector<uint16_t> probes_host;
};

class BinaryWeightLoader {
private:
    Gpu* gpu;
    std::string base_path;
    
    // Read metadata from .txt file
    WeightMetadata readMetadata(const std::string& meta_file) {
        std::ifstream file(meta_file);
        if (!file) {
            throw std::runtime_error("Cannot open metadata file: " + meta_file);
        }
        
        WeightMetadata meta;
        std::string line;
        
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            
            if (key == "buckets_shape:") {
                iss >> meta.buckets_shape.dim0 >> meta.buckets_shape.dim1 >> meta.buckets_shape.dim2;
            }
            else if (key == "stats_shape:") {
                iss >> meta.stats_shape.dim0 >> meta.stats_shape.dim1 >> meta.stats_shape.dim2;
            }
            else if (key == "probes_shape:") {
                iss >> meta.probes_shape.dim0 >> meta.probes_shape.dim1;
            }
            else if (key == "in_dim:") {
                iss >> meta.in_dim;
            }
            else if (key == "out_dim:") {
                iss >> meta.out_dim;
            }
            else if (key == "bsize:") {
                iss >> meta.bsize;
            }
        }
        
        // Validate
        if (meta.in_dim == 0 || meta.out_dim == 0) {
            throw std::runtime_error("Invalid metadata - missing dimensions");
        }
        
        return meta;
    }
    
    // Read raw binary file
    std::vector<uint16_t> readBinary(const std::string& bin_file, size_t expected_elements) {
        std::ifstream file(bin_file, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open binary file: " + bin_file);
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Validate size
        size_t expected_size = expected_elements * sizeof(uint16_t);
        if (file_size != expected_size) {
            std::ostringstream oss;
            oss << "File size mismatch: " << bin_file 
                << " has " << file_size << " bytes, expected " << expected_size;
            throw std::runtime_error(oss.str());
        }
        
        // Read data
        std::vector<uint16_t> data(expected_elements);
        file.read((char*)data.data(), file_size);
        
        if (!file) {
            throw std::runtime_error("Failed to read data from: " + bin_file);
        }
        
        return data;
    }
    
public:
    BinaryWeightLoader(Gpu* gpu, const std::string& base_path) 
        : gpu(gpu), base_path(base_path) {}
    
    // Load complete bucketized weight
    BucketizedWeight load(const std::string& weight_name) {
        // std::cout << "\n=== Loading Weight: " << weight_name << " ===" << std::endl;
        
        std::string prefix = base_path + "/" + weight_name;
        
        // Read metadata
        std::string meta_file = prefix + "_meta.txt";
        // std::cout << "Reading metadata: " << meta_file << std::endl;
        WeightMetadata meta = readMetadata(meta_file);
        
        // std::cout << "  Buckets shape: (" << meta.buckets_shape.dim0 << ", " 
        //           << meta.buckets_shape.dim1 << ", " << meta.buckets_shape.dim2 << ")" << std::endl;
        // std::cout << "  Stats shape: (" << meta.stats_shape.dim0 << ", " 
        //           << meta.stats_shape.dim1 << ", " << meta.stats_shape.dim2 << ")" << std::endl;
        // std::cout << "  Probes shape: (" << meta.probes_shape.dim0 << ", " 
        //           << meta.probes_shape.dim1 << ")" << std::endl;
        // std::cout << "  Dimensions: in=" << meta.in_dim << ", out=" << meta.out_dim 
        //           << ", bsize=" << meta.bsize << std::endl;
        
        // Read binary files
        std::string buckets_file = prefix + "_buckets.bin";
        std::string stats_file = prefix + "_stats.bin";
        std::string probes_file = prefix + "_probes.bin";
        
        // std::cout << "\nReading binary data..." << std::endl;
        
        std::vector<uint16_t> buckets = readBinary(buckets_file, meta.buckets_shape.total());
        // std::cout << "  âœ“ Buckets: " << buckets.size() << " elements (" 
        //           << (buckets.size() * 2) / (1024.0 * 1024.0) << " MB)" << std::endl;
        
        std::vector<uint16_t> stats = readBinary(stats_file, meta.stats_shape.total());
        // std::cout << "  âœ“ Stats: " << stats.size() << " elements ("
        //           << (stats.size() * 2) / (1024.0 * 1024.0) << " MB)" << std::endl;
        
        std::vector<uint16_t> probes = readBinary(probes_file, meta.probes_shape.total());
        // std::cout << "  âœ“ Probes: " << probes.size() << " elements ("
        //           << (probes.size() * 2) / (1024.0 * 1024.0) << " MB)" << std::endl;
        
        // DON'T upload to GPU here - let the caller do it
        // This avoids double allocation (loader creates buffers, then caller creates new ones)
        
        // Return weight structure with only host data
        BucketizedWeight weight;
        weight.buckets = nullptr;  // No GPU buffer from loader
        weight.stats = nullptr;
        weight.probes = nullptr;
        weight.meta = meta;
        weight.buckets_host = std::move(buckets);
        weight.stats_host = std::move(stats);
        weight.probes_host = std::move(probes);
        
        return weight;
    }
    
    // Helper: Load just for a specific layer and weight type
    BucketizedWeight loadLayerWeight(int layer, const std::string& weight_type) {
        std::ostringstream oss;
        oss << "layer" << layer << "_" << weight_type << "_proj_bucketmul_fp16";
        return load(oss.str());
    }
};

// Helper function to convert FP16 bits to float for debugging
inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF);
    
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            uint32_t result = sign;
            return *reinterpret_cast<float*>(&result);
        } else {
            // Subnormal: value = (-1)^sign * 2^-14 * (mantissa / 1024)
            //                  = (-1)^sign * mantissa * 2^-24
            float value = std::ldexp((float)mantissa, -24);
            return (sign != 0) ? -value : value;
        }
    } else if (exponent == 31) {
        // Inf or NaN
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    } else {
        // Normal number
        exponent = exponent - 15 + 127;
        uint32_t result = sign | (exponent << 23) | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    }
}

// Helper function to extract position from packed FP16
inline int extract_position(uint16_t packed_fp16) {
    return packed_fp16 & 0x000F;  // Lower 4 bits
}

// Helper function to extract value from packed FP16
inline float extract_value(uint16_t packed_fp16) {
    uint16_t value_bits = packed_fp16 & 0xFFF0;  // Clear lower 4 bits
    return fp16_to_float(value_bits);
}

#endif // BINARY_LOADER_HPP