cmake_minimum_required(VERSION 3.10)
project(effort_engine_opencl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCL REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Header files list
set(HEADERS
    gpu.hpp
    bucket_mul.hpp
    binary_loader.hpp    # NEW: Updated weight loader
    attention.hpp
    feedforward.hpp
    transformer_layer.hpp
    model.hpp
    tokenizer.hpp
)

# Common include directories for all targets
set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR})

# Common link libraries for all targets
set(COMMON_LIBS OpenCL::OpenCL nlohmann_json::nlohmann_json)

# ============================================================================
# Existing Executables
# ============================================================================


# Test: Minimal token generation (Layer 0 only)
add_executable(test_generation test_generation.cpp ${HEADERS})
target_link_libraries(test_generation ${COMMON_LIBS})
target_include_directories(test_generation PRIVATE ${COMMON_INCLUDES})

# Test: Cutoff binary search kernel (with visualization)
add_executable(test_cutoff_binary_search test_cutoff_binary_search.cpp ${HEADERS})
target_link_libraries(test_cutoff_binary_search ${COMMON_LIBS})
target_include_directories(test_cutoff_binary_search PRIVATE ${COMMON_INCLUDES})


# ============================================================================
# NEW: Weight Loading Test
# ============================================================================


# ============================================================================
# Kernel Files Setup
# ============================================================================

# Create kernels directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/kernels)

# Copy OpenCL kernel files
configure_file(${CMAKE_SOURCE_DIR}/bucketMul.cl 
               ${CMAKE_BINARY_DIR}/kernels/bucketMul.cl 
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/aux.cl 
               ${CMAKE_BINARY_DIR}/kernels/aux.cl 
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/matrix.cl 
               ${CMAKE_BINARY_DIR}/kernels/matrix.cl 
               COPYONLY)

# ============================================================================
# Installation Rules (Optional)
# ============================================================================

# Install executables
install(TARGETS 
    test_generation 
    test_cutoff_binary_search
    RUNTIME DESTINATION bin
)

# Install kernel files
install(FILES 
    ${CMAKE_SOURCE_DIR}/bucketMul.cl
    ${CMAKE_SOURCE_DIR}/aux.cl
    ${CMAKE_SOURCE_DIR}/matrix.cl
    DESTINATION share/kernels
)

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "=================================================================")
message(STATUS "Configuration Summary")
message(STATUS "=================================================================")
message(STATUS "  Project:             ${PROJECT_NAME}")
message(STATUS "  Build Type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenCL Found:        ${OpenCL_FOUND}")
message(STATUS "  OpenCL Include:      ${OpenCL_INCLUDE_DIRS}")
message(STATUS "  OpenCL Libraries:    ${OpenCL_LIBRARIES}")
message(STATUS "  nlohmann_json:       ${nlohmann_json_FOUND}")
message(STATUS "")
message(STATUS "Executables to build:")
message(STATUS "  - test_generation           (Token generation)")
message(STATUS "  - test_cutoff_binary_search (Cutoff kernel testing & visualization)")
message(STATUS "")
message(STATUS "Kernel files:")
message(STATUS "  - bucketMul.cl")
message(STATUS "  - aux.cl")
message(STATUS "  - matrix.cl")
message(STATUS "=================================================================")
message(STATUS "")

# ============================================================================
# Custom Targets for Common Operations
# ============================================================================

# Custom target: Run all tests
add_custom_target(run_tests
    COMMAND echo "Running test suite..."
    COMMAND test_tokenizer
    COMMAND test_weight_loading mistral_bucketmul 0
    COMMAND test_layer0_debug
    DEPENDS test_tokenizer test_weight_loading test_layer0_debug
    COMMENT "Running all tests"
)

# Custom target: Extract Python weights (if script is present)
add_custom_target(extract_weights
    COMMAND python3 ${CMAKE_SOURCE_DIR}/extract_npz_to_npy.py 
            --bucket-dir mistral_bucketmul --all
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Extracting Python NPZ weights to NPY format"
)

# Custom target: Clean weights
add_custom_target(clean_weights
    COMMAND find mistral_bucketmul -name "*_buckets.npy" -delete
    COMMAND find mistral_bucketmul -name "*_stats.npy" -delete
    COMMAND find mistral_bucketmul -name "*_probes.npy" -delete
    COMMAND find mistral_bucketmul -name "*_metadata.txt" -delete
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning extracted weight files"
)

# Help target
add_custom_target(help_targets
    COMMAND echo ""
    COMMAND echo "Available custom targets:"
    COMMAND echo "  make run_tests       - Run all test executables"
    COMMAND echo "  make extract_weights - Extract Python NPZ to NPY"
    COMMAND echo "  make clean_weights   - Remove extracted NPY files"
    COMMAND echo "  make help_targets    - Show this help"
    COMMAND echo ""
)