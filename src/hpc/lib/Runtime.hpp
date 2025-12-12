#pragma once

#include <cstddef>

extern "C" {

// dot
float hpc_dot_seq_f32(size_t n, const float *src1, const float *src2);
double hpc_dot_seq_f64(size_t n, const double *src1, const double *src2);
}
