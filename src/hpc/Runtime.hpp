#pragma once

#include <cstddef>

// C-compatible runtime functions that will be called from lowered LLVM code
extern "C" {

// AXPY:  Y = alpha * X + Y
void hpc_axpy_f32(size_t n, float *dst, const float *src, float alpha);
void hpc_axpy_f64(size_t n, double *dst, const double *src, double alpha);

// COPY: Y = X
void hpc_copy_f32(size_t n, float *dst, const float *src);
void hpc_copy_f64(size_t n, double *dst, const double *src);

// SCAL: X = alpha * X
void hpc_scal_f32(size_t n, float *dst, float alpha);
void hpc_scal_f64(size_t n, double *dst, double alpha);

// DOT: result = X · Y
float hpc_dot_f32(size_t n, const float *src1, const float *src2);
double hpc_dot_f64(size_t n, const double *src1, const double *src2);
}
