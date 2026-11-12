// bucket_mul_largecache.hpp - SIMPLE OPTIMIZATION
//
// Just increased DISPATCH_CACHE_SIZE from 64 to 256
// Update the local memory allocation in mul()
//
#ifndef BUCKET_MUL_LARGECACHE_HPP
#define BUCKET_MUL_LARGECACHE_HPP

#include "gpu.hpp"
#include <memory>
#include <cassert>

template<typename T>
class BufferWrapper : public MTLBufferable {
public:
    size_t count;
    size_t rows;
    size_t cols;
    size_t byte_size;
    Gpu* gpu_ref;
    
    BufferWrapper(Gpu* gpu, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE) 
        : MTLBufferable(gpu->createBuffer(size * sizeof(T), flags), 0),
          count(size), rows(size), cols(1), byte_size(size * sizeof(T)), gpu_ref(gpu) {}
    
    BufferWrapper(Gpu* gpu, const std::vector<size_t>& shape, cl_mem_flags flags = CL_MEM_READ_WRITE)
        : MTLBufferable(gpu->createBuffer(calculateSize(shape) * sizeof(T), flags), 0),
          count(calculateSize(shape)),
          rows(shape.size() > 0 ? shape[0] : 1),
          cols(shape.size() > 1 ? shape[1] : 1),
          byte_size(calculateSize(shape) * sizeof(T)),
          gpu_ref(gpu) {}
    
    void zero(Gpu* gpu) {
        T zero_val = T(0);
        clEnqueueFillBuffer(gpu->getQueue(), buffer, &zero_val, sizeof(T),
                            0, count * sizeof(T), 0, nullptr, nullptr);
    }
    
    T get(Gpu* gpu, size_t index) {
        gpu->eval();
        std::vector<T> data(count);
        gpu->readBuffer(buffer, count * sizeof(T), data.data());
        return data[index];
    }
    
    std::vector<T> getData(Gpu* gpu) {
        gpu->eval();
        std::vector<T> data(count);
        gpu->readBuffer(buffer, count * sizeof(T), data.data());
        return data;
    }
    
    void setData(Gpu* gpu, const std::vector<T>& data) {
        assert(data.size() == count);
        gpu->writeBuffer(buffer, count * sizeof(T), data.data());
    }
    
    ~BufferWrapper() {}
    
public:
    static size_t calculateSize(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        return size;
    }
};

class ScalarFloat : public BufferWrapper<float> {
public:
    ScalarFloat(Gpu* gpu, float value = 0.0f) 
        : BufferWrapper<float>(gpu, 1, CL_MEM_READ_WRITE) {
        std::vector<float> data = {value};
        gpu->writeBuffer(buffer, sizeof(float), data.data());
    }
    
    void zero(Gpu* gpu) {
        float zero_val = 0.0f;
        clEnqueueFillBuffer(gpu->getQueue(), buffer, &zero_val, sizeof(float),
                            0, sizeof(float), 0, nullptr, nullptr);
    }
    
    float getValue(Gpu* gpu) { return get(gpu, 0); }
};

class DynaVectorFloat : public BufferWrapper<float> {
public:
    BufferWrapper<uint32_t>* size;
    
    DynaVectorFloat(Gpu* gpu, const std::vector<size_t>& shape)
        : BufferWrapper<float>(gpu, shape, CL_MEM_READ_WRITE) {
        size = new BufferWrapper<uint32_t>(gpu, 1, CL_MEM_READ_WRITE);
        size->zero(gpu);
    }
    
    ~DynaVectorFloat() { delete size; }
};

class BucketMul {
public:
    static constexpr int PROBES_COUNT = 4096;
    static constexpr int MAX_DISPATCH_SIZE = 229376 * 2;
    static constexpr int MUL_GROUPS = 64;
    static constexpr int WAVEFRONT_SIZE = 64;
    
    // INCREASED from 64 to 256
    static constexpr int DISPATCH_CACHE_SIZE = 256;
    
    Gpu* gpu;
    
    DynaVectorFloat* dispatch;
    ScalarFloat* cutoff;
    BufferWrapper<int>* expertNo;
    BufferWrapper<uint32_t>* prevSize;
    BufferWrapper<float>* tmpMulVec;
    
    int cached_expert_no = -1;
    
    static BucketMul* shared(Gpu* g) {
        static BucketMul instance(g);
        return &instance;
    }
    
private:
    BucketMul(Gpu* g) : gpu(g) {
        dispatch = new DynaVectorFloat(gpu, {MAX_DISPATCH_SIZE * 2});
        cutoff = new ScalarFloat(gpu);
        expertNo = new BufferWrapper<int>(gpu, 1);
        prevSize = new BufferWrapper<uint32_t>(gpu, 1);
        prevSize->zero(gpu);
        tmpMulVec = new BufferWrapper<float>(gpu, {(size_t)MUL_GROUPS, 16384});
        
        std::vector<int> init_exp = {0};
        expertNo->setData(gpu, init_exp);
        cached_expert_no = 0;
    }
    
    ~BucketMul() {
        delete dispatch;
        delete cutoff;
        delete expertNo;
        delete prevSize;
        delete tmpMulVec;
    }

public:
    void calcDispatch(BufferWrapper<float>* input,
                      BufferWrapper<uint16_t>* stats,
                      BufferWrapper<uint16_t>* probes,
                      int expertNo_val,
                      int inSize,
                      int bucketCols,
                      int expertSize,
                      float effort) {
        
        if (expertNo_val != cached_expert_no) {
            std::vector<int> expNo_data = {expertNo_val};
            expertNo->setData(gpu, expNo_data);
            cached_expert_no = expertNo_val;
        }
        
        dispatch->size->zero(gpu);
        
        int quantile = (int)((double)(PROBES_COUNT - 1) * (1.0 - effort));
        
        cl_kernel cutoffKernel = gpu->getKernel("findCutoffBinarySearch");
        cl_int err;
        err = clSetKernelArg(cutoffKernel, 0, sizeof(cl_mem), &input->buffer);
        err |= clSetKernelArg(cutoffKernel, 1, sizeof(cl_mem), &probes->buffer);
        err |= clSetKernelArg(cutoffKernel, 2, sizeof(cl_mem), &expertNo->buffer);
        err |= clSetKernelArg(cutoffKernel, 3, sizeof(cl_mem), &cutoff->buffer);
        err |= clSetKernelArg(cutoffKernel, 4, sizeof(int), &quantile);
        err |= clSetKernelArg(cutoffKernel, 5, 1024 * sizeof(float), nullptr);

        size_t cutoff_global = 256;
        size_t cutoff_local = 256;
        gpu->enqueueKernel("findCutoffBinarySearch", cutoffKernel, 1, nullptr, &cutoff_global, &cutoff_local);
        
        int chunkSize = 4;
        int numThreads = stats->rows / chunkSize;
        size_t globalSize = ((numThreads + WAVEFRONT_SIZE - 1) / WAVEFRONT_SIZE) * WAVEFRONT_SIZE;
        
        gpu->deploy("prepareDispatch", 
            {input, stats, expertNo, cutoff, dispatch, dispatch->size},
            {chunkSize, inSize, bucketCols, expertSize}, 
            {},
            {},
            {globalSize},
            {WAVEFRONT_SIZE}
        );
    }
    
    void mul(BufferWrapper<uint16_t>* buckets, BufferWrapper<float>* output) {
    gpu->deploy("roundUp", 
        {dispatch->size, prevSize},
        {2048},
        {},
        {},
        {1},
        {1}
    );
    
    gpu->deploy("zeroRange32",
        {dispatch, prevSize, dispatch->size},
        {},
        {},
        {},
        {2048},
        {WAVEFRONT_SIZE}
    );
    
    size_t roundedCols = ((buckets->cols + WAVEFRONT_SIZE - 1) / WAVEFRONT_SIZE) * WAVEFRONT_SIZE;
    
    // KEY CHANGE: Allocate larger local memory (256 × 8 bytes = 2KB)
    cl_kernel mulKernel = gpu->getKernel("bucketMul");
    cl_int err;
    err = clSetKernelArg(mulKernel, 0, sizeof(cl_mem), &buckets->buffer);
    err |= clSetKernelArg(mulKernel, 1, sizeof(cl_mem), &dispatch->buffer);
    err |= clSetKernelArg(mulKernel, 2, sizeof(cl_mem), &tmpMulVec->buffer);
    err |= clSetKernelArg(mulKernel, 3, sizeof(cl_mem), &dispatch->size->buffer);
    int cols = buckets->cols;
    int groups = MUL_GROUPS;
    err |= clSetKernelArg(mulKernel, 4, sizeof(int), &cols);
    err |= clSetKernelArg(mulKernel, 5, sizeof(int), &groups);
    
    // INCREASED: 256 × sizeof(float2) = 256 × 8 = 2048 bytes
    err |= clSetKernelArg(mulKernel, 6, DISPATCH_CACHE_SIZE * sizeof(float) * 2, nullptr);
    
    size_t mulGlobal[2] = {roundedCols, (size_t)MUL_GROUPS};
    size_t mulLocal[2] = {(size_t)WAVEFRONT_SIZE, 1};
    gpu->enqueueKernel("bucketMul", mulKernel, 2, nullptr, mulGlobal, mulLocal);
    
    // bucketIntegrate kernel
    cl_kernel integrateKernel = gpu->getKernel("bucketIntegrate");
    err = clSetKernelArg(integrateKernel, 0, sizeof(cl_mem), &tmpMulVec->buffer);
    err |= clSetKernelArg(integrateKernel, 1, sizeof(cl_mem), &output->buffer);
    
    // FIX #1: Allocate proper local memory size (64 floats, not 1 float)
    err |= clSetKernelArg(integrateKernel, 2, WAVEFRONT_SIZE * sizeof(float), nullptr);
    
    // FIX #2: Correct Y dimension (don't divide by INTSTEPS twice!)
    size_t integrate_global[2] = {
        (size_t)WAVEFRONT_SIZE,     // X: one wavefront
        output->count / 4           // Y: each workgroup processes 4 outputs (INTSTEPS=4)
    };
    size_t integrate_local[2] = {
        (size_t)WAVEFRONT_SIZE,     // X: full wavefront
        1                            // Y: one row at a time
    };
    
    gpu->enqueueKernel("bucketIntegrate", integrateKernel, 2, nullptr, integrate_global, integrate_local);
}
    
    void fullMul(BufferWrapper<float>* input,
                 BufferWrapper<uint16_t>* stats,
                 BufferWrapper<uint16_t>* probes,
                 BufferWrapper<uint16_t>* buckets,
                 int expertNo_val,
                 int inSize,
                 int bucketCols,
                 int expertSize,
                 float effort,
                 BufferWrapper<float>* output) {
        
        calcDispatch(input, stats, probes, expertNo_val, inSize, bucketCols, expertSize, effort);
        mul(buckets, output);
    }
};

#endif