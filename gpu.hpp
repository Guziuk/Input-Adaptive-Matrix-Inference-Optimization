// gpu.hpp - DIAGNOSTIC VERSION with per-kernel timing
// UPDATED: Added enqueueKernel() helper for universal tracking
#ifndef GPU_HPP
#define GPU_HPP

#include <CL/cl.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <functional>
#include <chrono>
#include <algorithm>
#include <iomanip>


class MemoryTracker {
private:
    size_t total_vram;
    size_t allocated_vram;
    size_t safety_margin;
    
public:
    MemoryTracker(cl_device_id device) : allocated_vram(0), safety_margin(512ULL * 1024 * 1024) {
        cl_ulong mem_size;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, nullptr);
        total_vram = (size_t)mem_size;
        
        std::cout << "\n=== GPU Memory Info ===\n";
        std::cout << "Total VRAM: " << (total_vram / (1024.0 * 1024 * 1024)) << " GB\n";
        std::cout << "Safety margin: " << (safety_margin / (1024.0 * 1024 * 1024)) << " GB\n";
        std::cout << "Usable VRAM: " << ((total_vram - safety_margin) / (1024.0 * 1024 * 1024)) << " GB\n";
        std::cout << "========================\n\n";
    }
    
    bool can_allocate(size_t bytes) {
        size_t would_use = allocated_vram + bytes;
        size_t available = total_vram - safety_margin;
        return would_use <= available;
    }
    
    void track_allocation(size_t bytes) { allocated_vram += bytes; }
    void track_deallocation(size_t bytes) {
        if (bytes > allocated_vram) allocated_vram = 0;
        else allocated_vram -= bytes;
    }
    
    void print_status() {
        std::cout << "VRAM: " << (allocated_vram / (1024.0 * 1024 * 1024)) << " / " 
                  << ((total_vram - safety_margin) / (1024.0 * 1024 * 1024)) << " GB ("
                  << (int)(allocated_vram * 100.0f / (total_vram - safety_margin)) << "%)\n";
    }
    
    size_t get_allocated() const { return allocated_vram; }
    size_t get_available() const { return total_vram - safety_margin - allocated_vram; }
};

class MTLBufferable {
public:
    cl_mem buffer;
    size_t offsetBytes;
    
    MTLBufferable(cl_mem buf, size_t offset = 0) 
        : buffer(buf), offsetBytes(offset) {}
    
    virtual ~MTLBufferable() = default;
};

// Kernel timing info
struct KernelStats {
    int callCount = 0;
    double totalTimeMs = 0.0;
    double maxTimeMs = 0.0;
    size_t totalWorkItems = 0;
};

class Gpu {
public:
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    
    std::unordered_map<std::string, cl_kernel> globalKernels;
    
    bool warnOfEvals;
    bool captureON;
    
    std::vector<cl_event> pendingEvents;
    std::vector<std::string> pendingKernelNames;  // Track which kernel each event belongs to
    
    MemoryTracker* memTracker;
    
    // PROFILING COUNTERS
    int evalCount = 0;
    int blockingWriteCount = 0;
    int blockingReadCount = 0;
    double totalEvalTimeMs = 0.0;
    
    // PER-KERNEL PROFILING
    std::unordered_map<std::string, KernelStats> kernelStats;
    bool enableKernelProfiling = true;  // Set to true for detailed timing
    
public:
    Gpu() : warnOfEvals(false), captureON(false), memTracker(nullptr), 
            evalCount(0), blockingWriteCount(0), blockingReadCount(0), totalEvalTimeMs(0.0) {
        cl_int err;
        
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        checkError(err, "Failed to get platform");
        
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        }
        checkError(err, "Failed to get device");
        
        char deviceName[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        std::cout << "Using device: " << deviceName << std::endl;
        
        // Get max work group size
        size_t maxWorkGroupSize;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr);
        std::cout << "Max work group size: " << maxWorkGroupSize << std::endl;
        
        // Get max compute units
        cl_uint computeUnits;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, nullptr);
        std::cout << "Compute units: " << computeUnits << std::endl;
        
        // Get preferred work group size multiple (wavefront size for AMD)
        // This requires a kernel, so we'll check after compilation
        
        char extensions[4096];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, nullptr);
        if (strstr(extensions, "cl_khr_fp16") != nullptr) {
            std::cout << "FP16 support: YES" << std::endl;
        } else {
            std::cout << "FP16 support: NO" << std::endl;
        }
        
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        checkError(err, "Failed to create context");
        
        // IMPORTANT: Enable profiling in command queue
        commandQueue = clCreateCommandQueue(context, device, 
                                           CL_QUEUE_PROFILING_ENABLE, &err);
        checkError(err, "Failed to create command queue");
        
        memTracker = new MemoryTracker(device);
        
        loadKernels();
    }
    
    ~Gpu() {
        // Print kernel profiling results
        printKernelStats();
        
        for (auto& kv : globalKernels) {
            clReleaseKernel(kv.second);
        }
        
        for (auto event : pendingEvents) {
            clReleaseEvent(event);
        }
        
        if (program) clReleaseProgram(program);
        if (commandQueue) clReleaseCommandQueue(commandQueue);
        if (context) clReleaseContext(context);
        
        if (memTracker) {
            std::cout << "\n========== OVERALL PROFILING ==========\n";
            std::cout << "Total eval() calls: " << evalCount << std::endl;
            std::cout << "Total blocking writes: " << blockingWriteCount << std::endl;
            std::cout << "Total blocking reads: " << blockingReadCount << std::endl;
            std::cout << "Total eval time: " << totalEvalTimeMs << " ms\n";
            std::cout << "Avg eval time: " << (evalCount > 0 ? totalEvalTimeMs/evalCount : 0) << " ms\n";
            std::cout << "========================================\n";
            memTracker->print_status();
            delete memTracker;
        }
    }
    
    void printKernelStats() {
        if (kernelStats.empty()) return;
        
        std::cout << "\n========== PER-KERNEL PROFILING ==========\n";
        std::cout << std::setw(25) << "Kernel" 
                  << std::setw(10) << "Calls"
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(12) << "Max (ms)"
                  << std::setw(15) << "% of Total"
                  << "\n";
        std::cout << std::string(92, '-') << "\n";
        
        // Sort by total time
        std::vector<std::pair<std::string, KernelStats>> sorted(kernelStats.begin(), kernelStats.end());
        std::sort(sorted.begin(), sorted.end(), 
                  [](const auto& a, const auto& b) { return a.second.totalTimeMs > b.second.totalTimeMs; });
        
        double totalKernelTime = 0;
        for (const auto& kv : sorted) {
            totalKernelTime += kv.second.totalTimeMs;
        }
        
        for (const auto& kv : sorted) {
            const auto& name = kv.first;
            const auto& stats = kv.second;
            double avgTime = stats.callCount > 0 ? stats.totalTimeMs / stats.callCount : 0;
            double pctTotal = totalKernelTime > 0 ? (stats.totalTimeMs / totalKernelTime) * 100 : 0;
            
            std::cout << std::setw(25) << name
                      << std::setw(10) << stats.callCount
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats.totalTimeMs
                      << std::setw(15) << std::fixed << std::setprecision(4) << avgTime
                      << std::setw(12) << std::fixed << std::setprecision(4) << stats.maxTimeMs
                      << std::setw(14) << std::fixed << std::setprecision(1) << pctTotal << "%"
                      << "\n";
        }
        
        std::cout << std::string(92, '-') << "\n";
        std::cout << std::setw(25) << "TOTAL"
                  << std::setw(10) << ""
                  << std::setw(15) << std::fixed << std::setprecision(2) << totalKernelTime
                  << "\n";
        std::cout << "==========================================\n";
    }
    
    cl_mem createBuffer(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE) {
        if (!memTracker->can_allocate(size)) {
            throw std::runtime_error("VRAM allocation failed - insufficient memory!");
        }
        
        cl_int err;
        cl_mem buffer = clCreateBuffer(context, flags, size, nullptr, &err);
        checkError(err, "Failed to create buffer");
        
        memTracker->track_allocation(size);
        
        return buffer;
    }
    
    cl_mem createBuffer(size_t size, void* data, cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR) {
        if (!memTracker->can_allocate(size)) {
            throw std::runtime_error("VRAM allocation failed - insufficient memory!");
        }
        
        cl_int err;
        cl_mem buffer = clCreateBuffer(context, flags, size, data, &err);
        checkError(err, "Failed to create buffer with data");
        
        memTracker->track_allocation(size);
        
        return buffer;
    }
    
    void releaseBuffer(cl_mem buffer, size_t size) {
        if (buffer) {
            clReleaseMemObject(buffer);
            memTracker->track_deallocation(size);
        }
    }
    
    // =========================================================================
    // NEW: Universal kernel enqueue with tracking
    // Use this instead of raw clEnqueueNDRangeKernel to track ALL kernels
    // =========================================================================
    cl_int enqueueKernel(const std::string& kernelName,
                         cl_kernel kernel,
                         cl_uint workDim,
                         const size_t* globalWorkOffset,
                         const size_t* globalWorkSize,
                         const size_t* localWorkSize) {
        cl_event event;
        cl_int err = clEnqueueNDRangeKernel(commandQueue, kernel, workDim,
                                            globalWorkOffset, globalWorkSize, localWorkSize,
                                            0, nullptr, &event);
        if (err == CL_SUCCESS) {
            pendingEvents.push_back(event);
            pendingKernelNames.push_back(kernelName);
        }
        return err;
    }
    
    // Convenience overload: 1D kernel
    cl_int enqueueKernel1D(const std::string& kernelName,
                           cl_kernel kernel,
                           size_t globalSize,
                           size_t localSize = 0) {
        size_t* local = (localSize > 0) ? &localSize : nullptr;
        return enqueueKernel(kernelName, kernel, 1, nullptr, &globalSize, local);
    }
    
    // Convenience overload: 2D kernel
    cl_int enqueueKernel2D(const std::string& kernelName,
                           cl_kernel kernel,
                           size_t globalX, size_t globalY,
                           size_t localX = 0, size_t localY = 0) {
        size_t global[2] = {globalX, globalY};
        size_t local[2] = {localX, localY};
        size_t* localPtr = (localX > 0 && localY > 0) ? local : nullptr;
        return enqueueKernel(kernelName, kernel, 2, nullptr, global, localPtr);
    }
    
    // Convenience overload: 3D kernel
    cl_int enqueueKernel3D(const std::string& kernelName,
                           cl_kernel kernel,
                           size_t globalX, size_t globalY, size_t globalZ,
                           size_t localX = 0, size_t localY = 0, size_t localZ = 0) {
        size_t global[3] = {globalX, globalY, globalZ};
        size_t local[3] = {localX, localY, localZ};
        size_t* localPtr = (localX > 0 && localY > 0 && localZ > 0) ? local : nullptr;
        return enqueueKernel(kernelName, kernel, 3, nullptr, global, localPtr);
    }
    
    void deploy(const std::string& kernelName,
                const std::vector<MTLBufferable*>& buffers,
                const std::vector<int>& ints = {},
                const std::vector<uint16_t>& float16s = {},
                const std::vector<float>& floats = {},
                size_t threadCount = 0,
                const std::vector<size_t>& threadGroupSize = {32, 1, 1}) {
        
        std::vector<size_t> tc = {threadCount};
        deploy(kernelName, buffers, ints, float16s, floats, tc, threadGroupSize);
    }
    
    void deploy(const std::string& kernelName,
                const std::vector<MTLBufferable*>& buffers,
                const std::vector<int>& ints,
                const std::vector<uint16_t>& float16s,
                const std::vector<float>& floats,
                const std::vector<size_t>& threadCount,
                const std::vector<size_t>& threadGroupSize = {32, 1, 1}) {
        
        if (globalKernels.find(kernelName) == globalKernels.end()) {
            std::cerr << "Warning: Kernel " << kernelName << " not found." << std::endl;
            makeFunction(kernelName);
        }
        
        cl_kernel kernel = globalKernels[kernelName];
        cl_int err;
        
        int idx = 0;
        for (const auto& b : buffers) {
            err = clSetKernelArg(kernel, idx, sizeof(cl_mem), &b->buffer);
            checkError(err, "Failed to set buffer argument");
            idx++;
        }
        
        for (const auto& i : ints) {
            err = clSetKernelArg(kernel, idx, sizeof(int), &i);
            checkError(err, "Failed to set int argument");
            idx++;
        }
        
        for (const auto& f16 : float16s) {
            err = clSetKernelArg(kernel, idx, sizeof(uint16_t), &f16);
            checkError(err, "Failed to set float16 argument");
            idx++;
        }
        
        for (const auto& f : floats) {
            err = clSetKernelArg(kernel, idx, sizeof(float), &f);
            checkError(err, "Failed to set float argument");
            idx++;
        }
        
        size_t globalWorkSize[3];
        size_t localWorkSize[3];
        cl_uint workDim = threadCount.size();
        
        for (size_t i = 0; i < 3; i++) {
            if (i < threadCount.size()) {
                globalWorkSize[i] = threadCount[i];
            } else {
                globalWorkSize[i] = 1;
            }
            
            if (i < threadGroupSize.size()) {
                localWorkSize[i] = threadGroupSize[i];
            } else {
                localWorkSize[i] = 1;
            }
        }
        
        // Use enqueueKernel for tracking
        err = enqueueKernel(kernelName, kernel, workDim, nullptr, globalWorkSize, localWorkSize);
        checkError(err, "Failed to enqueue kernel");
    }
    
    void eval(bool noWarn = false) {
        auto start = std::chrono::high_resolution_clock::now();
        
        evalCount++;
        
        if (warnOfEvals && !noWarn) {
            std::cout << "warn: EVAL #" << evalCount << std::endl;
        }
        
        cl_int err = clFinish(commandQueue);
        checkError(err, "Failed to finish command queue");
        
        // Collect kernel timing info
        if (enableKernelProfiling) {
            for (size_t i = 0; i < pendingEvents.size(); i++) {
                cl_event event = pendingEvents[i];
                const std::string& kernelName = pendingKernelNames[i];
                
                cl_ulong startTime, endTime;
                err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
                                              sizeof(cl_ulong), &startTime, nullptr);
                err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                               sizeof(cl_ulong), &endTime, nullptr);
                
                if (err == CL_SUCCESS) {
                    double timeMs = (endTime - startTime) / 1e6;  // ns to ms
                    
                    auto& stats = kernelStats[kernelName];
                    stats.callCount++;
                    stats.totalTimeMs += timeMs;
                    if (timeMs > stats.maxTimeMs) {
                        stats.maxTimeMs = timeMs;
                    }
                }
                
                clReleaseEvent(event);
            }
        } else {
            for (auto event : pendingEvents) {
                clReleaseEvent(event);
            }
        }
        
        pendingEvents.clear();
        pendingKernelNames.clear();
        
        auto end = std::chrono::high_resolution_clock::now();
        totalEvalTimeMs += std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void copyBuffer(MTLBufferable* src, MTLBufferable* dst, size_t size) {
        cl_int err = clEnqueueCopyBuffer(commandQueue,
                                        src->buffer, dst->buffer,
                                        src->offsetBytes, dst->offsetBytes,
                                        size, 0, nullptr, nullptr);
        checkError(err, "Failed to copy buffer");
    }
    
    void readBuffer(cl_mem buffer, size_t size, void* hostPtr) {
        blockingReadCount++;
        cl_int err = clEnqueueReadBuffer(commandQueue, buffer, CL_TRUE,
                                        0, size, hostPtr, 0, nullptr, nullptr);
        checkError(err, "Failed to read buffer");
    }
    
    void writeBuffer(cl_mem buffer, size_t size, const void* hostPtr) {
        blockingWriteCount++;
        cl_int err = clEnqueueWriteBuffer(commandQueue, buffer, CL_TRUE,
                                         0, size, hostPtr, 0, nullptr, nullptr);
        checkError(err, "Failed to write buffer");
    }
    
    cl_device_id getDevice() const { return device; }
    cl_context getContext() const { return context; }
    cl_command_queue getQueue() const { return commandQueue; }
    
    cl_kernel getKernel(const std::string& name) {
        auto it = globalKernels.find(name);
        if (it == globalKernels.end()) {
            throw std::runtime_error("Kernel not found: " + name);
        }
        return it->second;
    }
    
    void concurrent(const std::vector<std::function<void()>>& closures) {
        for (const auto& closure : closures) {
            closure();
        }
        // NO eval() here
    }
    
    void startCapture(bool cond = true) {
        if (!cond) return;
        if (captureON) return;
        eval();
        captureON = true;
    }
    
    void stopCapture() {
        if (captureON) {
            eval();
            captureON = false;
        }
    }

public:
    void loadKernels() {
        std::vector<std::string> kernelFiles = {
            "kernels/bucketMul.cl",
            "kernels/aux.cl",
            "kernels/matrix.cl",
            "kernels/bucketMul_diagnostic.cl"
        };
        
        std::string allSource;
        for (const auto& filename : kernelFiles) {
            std::ifstream file(filename);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                allSource += buffer.str() + "\n";
            } else {
                std::cerr << "Warning: Could not open " << filename << std::endl;
            }
        }
        
        if (allSource.empty()) {
            throw std::runtime_error("No kernel source files found!");
        }
        
        const char* source = allSource.c_str();
        size_t sourceSize = allSource.length();
        
        cl_int err;
        program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &err);
        checkError(err, "Failed to create program");
        
        const char* buildOptions = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
        err = clBuildProgram(program, 1, &device, buildOptions, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            size_t logSize;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "Build log:\n" << log.data() << std::endl;
            checkError(err, "Failed to build program");
        }
        
        std::vector<std::string> functionNames = {
            "memcpy32", "rmsNorm32", "halfToFloat", "mulVec32by16", "basicMul",
            "repeat4x32", "mulComplex32_mx", "rope_mx", "compute_attn_scores", "dotSetScore32", "zero32", "sum_of_exps32",
            "softmax_add32", "sumScores32", "add32", "floatToHalf",
            "basicBitonicSort", "getVal", "bucketMul",
            "silu32", "mulScalar32x32", "memcpy16", "memcpyBig16", "touch",
            "dotSetScore2", "findCutoff", "roundUp", "setVal",
            "cosinePrecalc32", "cosineCalc32",
            "prepareDispatch", "prepareDispatchQ4", "bucketMulQ4",
            "bucketIntegrate", "zeroRange32", "findCutoff32", "calcOutliers",
            "fetchRow16to32", "matmul16x32", "mul16x32","bucketMulAdaptive","bucketMulDiagnostic",
            "sum_of_exps32_mx", "softmax_add32_mx","findCutoffBinarySearch","bucketMulSparse", "rmsNorm32fast"
        };
        
        for (const auto& fname : functionNames) {
            makeFunction(fname);
        }
        
        std::cout << "Loaded " << globalKernels.size() << " kernels successfully" << std::endl;
    }
    
    void makeFunction(const std::string& fname) {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, fname.c_str(), &err);
        
        if (err != CL_SUCCESS) {
            std::cerr << "Warning: Failed to create kernel '" << fname << "'" << std::endl;
            return;
        }
        
        globalKernels[fname] = kernel;
    }
    
    void checkError(cl_int err, const char* message) {
        if (err != CL_SUCCESS) {
            std::stringstream ss;
            ss << message << " (error code: " << err << ")";
            throw std::runtime_error(ss.str());
        }
    }
};

#endif