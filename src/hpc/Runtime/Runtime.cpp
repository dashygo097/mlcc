#include "../Runtime.hpp"
#include <hpc.hh>

extern "C" {

void hpc_axpy_f32(size_t n, float *dst, const float *src, float alpha) {
  hpc::l1::details::axpy_seq(n, dst, src, alpha);
}

void hpc_axpy_f64(size_t n, double *dst, const double *src, double alpha) {
  hpc::l1::details::axpy_seq(n, dst, src, alpha);
}

void hpc_copy_f32(size_t n, float *dst, const float *src) {
  hpc::l1::details::copy_seq(n, dst, src);
}

void hpc_copy_f64(size_t n, double *dst, const double *src) {
  hpc::l1::details::copy_seq(n, dst, src);
}

void hpc_scal_f32(size_t n, float *dst, float alpha) {
  hpc::l1::details::scal_seq(n, dst, alpha);
}

void hpc_scal_f64(size_t n, double *dst, double alpha) {
  hpc::l1::details::scal_seq(n, dst, alpha);
}

float hpc_dot_f32(size_t n, const float *src1, const float *src2) {
  return hpc::l1::details::dot_seq(n, src1, src2);
}

double hpc_dot_f64(size_t n, const double *src1, const double *src2) {
  return hpc::l1::details::dot_seq(n, src1, src2);
}

} // extern "C"
