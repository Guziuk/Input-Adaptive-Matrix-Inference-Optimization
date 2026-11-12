#ifndef NPZ_LOADER_HPP
#define NPZ_LOADER_HPP

#include "gpu.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>

// Simple NPY header parser
struct NpyHeader {
    std::vector<size_t> shape;
    std::string dtype;
    size_t header_size;
};

NpyHeader parseNpyHeader(std::ifstream& file) {
    NpyHeader header;
    
    // Read magic bytes
    char magic[6];
    file.read(magic, 6);
    if (strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Not a valid NPY file");
    }
    
    // Read version
    uint8_t major, minor;
    file.read((char*)&major, 1);
    file.read((char*)&minor, 1);
    
    // Read header length
    uint16_t header_len;
    file.read((char*)&header_len, 2);
    
    // Read header string
    std::vector<char> header_str(header_len);
    file.read(header_str.data(), header_len);
    
    std::string hdr(header_str.begin(), header_str.end());
    
    // Parse shape - look for 'shape': (...)
    size_t shape_pos = hdr.find("'shape':");
    if (shape_pos != std::string::npos) {
        size_t start = hdr.find('(', shape_pos);
        size_t end = hdr.find(')', shape_pos);
        std::string shape_str = hdr.substr(start + 1, end - start - 1);
        
        // Parse comma-separated dimensions
        size_t pos = 0;
        while (pos < shape_str.size()) {
            // Skip whitespace and commas
            while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) {
                pos++;
            }
            
            if (pos >= shape_str.size()) break;
            
            // Read number
            size_t end_num = pos;
            while (end_num < shape_str.size() && isdigit(shape_str[end_num])) {
                end_num++;
            }
            
            if (end_num > pos) {
                std::string num_str = shape_str.substr(pos, end_num - pos);
                header.shape.push_back(std::stoull(num_str));
                pos = end_num;
            } else {
                pos++;
            }
        }
    }
    
    // Parse dtype
    size_t dtype_pos = hdr.find("'descr':");
    if (dtype_pos != std::string::npos) {
        size_t start = hdr.find('\'', dtype_pos + 8) + 1;
        size_t end = hdr.find('\'', start);
        header.dtype = hdr.substr(start, end - start);
    }
    
    header.header_size = 6 + 2 + 2 + header_len;
    
    return header;
}

// Load FP16 array from NPY file
std::vector<uint16_t> loadNpyFP16(const std::string& filename, std::vector<size_t>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    
    NpyHeader header = parseNpyHeader(file);
    shape = header.shape;
    
    // Calculate total elements
    size_t total = 1;
    for (auto dim : shape) {
        total *= dim;
    }
    
    // Read data
    std::vector<uint16_t> data(total);
    file.read((char*)data.data(), total * sizeof(uint16_t));
    
    std::cout << "Loaded " << filename << ": shape=(";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "), elements=" << total << std::endl;
    
    return data;
}

// Complete weight structure
struct BucketizedWeight {
    cl_mem buckets;
    cl_mem stats;
    cl_mem probes;
    
    std::vector<size_t> buckets_shape;
    std::vector<size_t> stats_shape;
    std::vector<size_t> probes_shape;
    
    size_t in_dim;
    size_t out_dim;
    size_t bsize;
    
    std::vector<uint16_t> buckets_host;
    std::vector<uint16_t> stats_host;
    std::vector<uint16_t> probes_host;
};

class NpzLoader {
private:
    Gpu* gpu;
    std::string base_path;
    
public:
    NpzLoader(Gpu* gpu, const std::string& base_path)
        : gpu(gpu), base_path(base_path) {}
    
    BucketizedWeight load(const std::string& npz_file) {
        std::cout << "\n=== Loading from NPZ: " << npz_file << " ===" << std::endl;
        
        // For NPZ, we need to extract to NPY first
        // Or load NPY files directly if already extracted
        
        std::string base = base_path + "/" + npz_file;
        
        // Try loading individual NPY files
        std::string buckets_file = base + "_buckets.npy";
        std::string stats_file = base + "_stats.npy";
        std::string probes_file = base + "_probes.npy";
        
        BucketizedWeight weight;
        
        // Load arrays
        // std::cout << "Loading buckets..." << std::endl;
        weight.buckets_host = loadNpyFP16(buckets_file, weight.buckets_shape);
        
        // std::cout << "Loading stats..." << std::endl;
        weight.stats_host = loadNpyFP16(stats_file, weight.stats_shape);
        
        // std::cout << "Loading probes..." << std::endl;
        weight.probes_host = loadNpyFP16(probes_file, weight.probes_shape);
        
        // Extract dimensions
        if (weight.buckets_shape.size() == 3) {
            weight.in_dim = weight.buckets_shape[1] / 16;
            weight.out_dim = weight.buckets_shape[2] * 16;
        } else {
            throw std::runtime_error("Invalid buckets shape");
        }
        weight.bsize = 16;
        
        // std::cout << "\nDimensions:" << std::endl;
        // std::cout << "  in_dim: " << weight.in_dim << std::endl;
        // std::cout << "  out_dim: " << weight.out_dim << std::endl;
        // std::cout << "  bsize: " << weight.bsize << std::endl;
        
        // Upload to GPU
        // std::cout << "\nUploading to GPU..." << std::endl;
        
        weight.buckets = gpu->createBuffer(
            weight.buckets_host.size() * sizeof(uint16_t),
            weight.buckets_host.data(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
        );
        std::cout << "  ✓ Buckets" << std::endl;
        
        weight.stats = gpu->createBuffer(
            weight.stats_host.size() * sizeof(uint16_t),
            weight.stats_host.data(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
        );
        std::cout << "  ✓ Stats" << std::endl;
        
        weight.probes = gpu->createBuffer(
            weight.probes_host.size() * sizeof(uint16_t),
            weight.probes_host.data(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
        );
        std::cout << "  ✓ Probes" << std::endl;
        
        std::cout << "\n✓ Loading complete!" << std::endl;
        
        return weight;
    }
    
    BucketizedWeight loadLayerWeight(int layer, const std::string& weight_type) {
        std::ostringstream oss;
        oss << "layer" << layer << "_" << weight_type << "_proj_bucketmul_fp16";
        return load(oss.str());
    }
};

// Helper functions (same as before)
inline float fp16_to_float(uint16_t bits) {
    uint32_t sign = (bits & 0x8000) << 16;
    uint32_t exponent = (bits & 0x7C00) >> 10;
    uint32_t mantissa = (bits & 0x03FF);
    
    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            return *reinterpret_cast<float*>(&result);
        }
        return 0.0f;
    } else if (exponent == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    } else {
        exponent = exponent - 15 + 127;
        uint32_t result = sign | (exponent << 23) | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    }
}

inline int extract_position(uint16_t packed_fp16) {
    return packed_fp16 & 0x000F;
}

inline float extract_value(uint16_t packed_fp16) {
    uint16_t value_bits = packed_fp16 & 0xFFF0;
    return fp16_to_float(value_bits);
}

#endif // NPZ_LOADER_HPP