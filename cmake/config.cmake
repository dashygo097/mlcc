# Build configuration options

# Compilation options
set(USE_CCACHE ON)

# Build options
set(ENABLE_TOOLS_BUILDING ON)
set(ENABLE_TESTING ON)

# HPC library backend configuration
set(HPC_ENABLE_SIMD ON CACHE BOOL "Enable SIMD in HPC library")
set(HPC_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP in HPC library")
set(HPC_ENABLE_MPI ON CACHE BOOL "Enable MPI in HPC library")
set(HPC_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA in HPC library")
set(HPC_ENABLE_ACCELERATE OFF CACHE BOOL "Enable Accelerate framework in HPC library (macOS)")
set(HPC_USE_HIGH_LEVEL_OPTIMIZATIONS ON CACHE BOOL "Enable high-level optimizations in HPC")
