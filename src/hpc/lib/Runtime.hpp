#pragma once

#include <cstddef>

extern "C" {
// DOT
float hpc_dot_f32(size_t n, const float *src1, const float *src2);
double hpc_dot_f64(size_t n, const double *src1, const double *src2);
}
