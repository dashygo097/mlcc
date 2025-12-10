#pragma once

#include <cstddef>

extern "C" {

// AXPY
void hpc_axpy_seq_f32(size_t n, float *dst, const float *src, float alpha);
void hpc_axpy_seq_f64(size_t n, double *dst, const double *src, double alpha);

// COPY
void hpc_copy_seq_f32(size_t n, float *dst, const float *src);
void hpc_copy_seq_f64(size_t n, double *dst, const double *src);

// SCAL
void hpc_scal_seq_f32(size_t n, float *dst, float alpha);
void hpc_scal_seq_f64(size_t n, double *dst, double alpha);

// DOT
float hpc_dot_seq_f32(size_t n, const float *src1, const float *src2);
double hpc_dot_seq_f64(size_t n, const double *src1, const double *src2);
}
