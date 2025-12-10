message(STATUS "=== HPC Dialect Configuration ===")

# Backend Options
set(ENABLE_SIMD ON)
set(ENABLE_OPENMP ON)
set(ENABLE_MPI OFF)
set(ENABLE_CUDA OFF)

# BLAS/LAPACK Options
set(ENABLE_ACCELERATE OFF)

# Compilation Options
set(USE_HIGH_LEVEL_OPTIMIZATIONS ON)
