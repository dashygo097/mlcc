# Find and configure all external dependencies

# --- 3rdparty libraries (always included) ---
include_directories(3rdparty)

# Add googletest and benchmark for MLCC tests 
if(ENABLE_TESTING)
  if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/googletest")
    add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
  endif()
  if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/benchmark")
    add_subdirectory(3rdparty/benchmark EXCLUDE_FROM_ALL)
  endif()
endif()

# --- HPC library ---
if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/hpc")
  message(STATUS "Building HPC BLAS library from 3rdparty/hpc")
  
  include(ExternalProject)
  
  # HPC configuration from MLCC config
  set(HPC_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_STANDARD=17
    
    # Use MLCC's configuration variables
    -DENABLE_SIMD=${HPC_ENABLE_SIMD}
    -DENABLE_OPENMP=${HPC_ENABLE_OPENMP}
    -DENABLE_MPI=${HPC_ENABLE_MPI}
    -DENABLE_CUDA=${HPC_ENABLE_CUDA}
    -DENABLE_ACCELERATE=${HPC_ENABLE_ACCELERATE}
    -DUSE_HIGH_LEVEL_OPTIMIZATIONS=${HPC_USE_HIGH_LEVEL_OPTIMIZATIONS}
    
    # Python - always off for MLCC
    -DENABLE_PYBIND11=OFF
    -DENABLE_PYTORCH=OFF
    
    # Testing/Benchmarking - always off when used as dependency
    -DENABLE_TESTING=OFF
    -DENABLE_BENCHMARKING=OFF
    
    -DUSE_CCACHE=${USE_CCACHE}
  )
  
  ExternalProject_Add(hpc_ext
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/hpc
    BINARY_DIR ${CMAKE_BINARY_DIR}/3rdparty/hpc
    CMAKE_ARGS ${HPC_CMAKE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target hpc -j
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/3rdparty/hpc/lib/libhpc.a
  )
  
  # Create imported library target
  add_library(hpc STATIC IMPORTED GLOBAL)
  set_target_properties(hpc PROPERTIES
    IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/3rdparty/hpc/lib/libhpc.a
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/3rdparty/hpc/include"
  )
  
  add_dependencies(hpc hpc_ext)
  include_directories(${CMAKE_SOURCE_DIR}/3rdparty/hpc/include)
  
  set(HAS_HPC TRUE)
  set(HPC_LIBRARY hpc)
  
  message(STATUS "HPC library configuration:")
  message(STATUS "  - SIMD: ${HPC_ENABLE_SIMD}")
  message(STATUS "  - OpenMP: ${HPC_ENABLE_OPENMP}")
  message(STATUS "  - MPI: ${HPC_ENABLE_MPI}")
  message(STATUS "  - CUDA: ${HPC_ENABLE_CUDA}")
  message(STATUS "  - Accelerate: ${HPC_ENABLE_ACCELERATE}")
  message(STATUS "  - High-level opts: ${HPC_USE_HIGH_LEVEL_OPTIMIZATIONS}")
else()
  message(FATAL_ERROR "HPC library not found in 3rdparty/hpc")
endif()

# --- LLVM ---
find_package(LLVM REQUIRED CONFIG)
include(AddLLVM)
include(TableGen)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig. cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})

# LLVM definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
set(HAS_LLVM TRUE)

llvm_map_components_to_libnames(llvm_libs support core irreader)

# --- MLIR ---
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/include)
set(HAS_MLIR TRUE)
