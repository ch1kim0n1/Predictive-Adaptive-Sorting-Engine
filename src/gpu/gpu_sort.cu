#include "gpu_api.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#if defined(PASE_GPU_SORT_USE_CUB)
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#endif

#include <complex>
#include <cstring>
#include <vector>

namespace pase {

bool gpu_sort_int_available() {
  int nd = 0;
  if (cudaGetDeviceCount(&nd) != cudaSuccess) {
    return false;
  }
  return nd > 0;
}

bool gpu_sort_device_available() { return gpu_sort_int_available(); }

namespace {

bool check(cudaError_t e) { return e == cudaSuccess; }

constexpr int kMinPracticalGpuSortN = 8192;

struct LexComplexF {
  float r, i;
  __host__ __device__ bool operator<(const LexComplexF& o) const {
    if (r < o.r) {
      return true;
    }
    if (o.r < r) {
      return false;
    }
    return i < o.i;
  }
};

struct LexComplexD {
  double r, i;
  __host__ __device__ bool operator<(const LexComplexD& o) const {
    if (r < o.r) {
      return true;
    }
    if (o.r < r) {
      return false;
    }
    return i < o.i;
  }
};

template <typename T>
bool gpu_sort_thrust_t(T* host, int n) {
  if (n <= 1) {
    return true;
  }
  if (!gpu_sort_int_available()) {
    return false;
  }
  if (n < kMinPracticalGpuSortN) {
    return false;
  }
  const size_t bytes = static_cast<size_t>(n) * sizeof(T);
  T* d = nullptr;
  if (!check(cudaMalloc(reinterpret_cast<void**>(&d), bytes))) {
    return false;
  }
  if (!check(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice))) {
    cudaFree(d);
    return false;
  }
  thrust::sort(thrust::device, d, d + n);
  if (!check(cudaGetLastError())) {
    cudaFree(d);
    return false;
  }
  if (!check(cudaMemcpy(host, d, bytes, cudaMemcpyDeviceToHost))) {
    cudaFree(d);
    return false;
  }
  cudaFree(d);
  return true;
}

}  // namespace

bool gpu_sort_float(float* host, int n) { return gpu_sort_thrust_t(host, n); }

bool gpu_sort_double(double* host, int n) { return gpu_sort_thrust_t(host, n); }

bool gpu_sort_complex_float(std::complex<float>* host, int n) {
  if (n <= 1) {
    return true;
  }
  std::vector<LexComplexF> tmp(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    tmp[static_cast<size_t>(i)].r = host[i].real();
    tmp[static_cast<size_t>(i)].i = host[i].imag();
  }
  if (!gpu_sort_thrust_t(tmp.data(), n)) {
    return false;
  }
  for (int i = 0; i < n; ++i) {
    const auto& c = tmp[static_cast<size_t>(i)];
    host[i] = std::complex<float>(c.r, c.i);
  }
  return true;
}

bool gpu_sort_complex_double(std::complex<double>* host, int n) {
  if (n <= 1) {
    return true;
  }
  std::vector<LexComplexD> tmp(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    tmp[static_cast<size_t>(i)].r = host[i].real();
    tmp[static_cast<size_t>(i)].i = host[i].imag();
  }
  if (!gpu_sort_thrust_t(tmp.data(), n)) {
    return false;
  }
  for (int i = 0; i < n; ++i) {
    const auto& c = tmp[static_cast<size_t>(i)];
    host[i] = std::complex<double>(c.r, c.i);
  }
  return true;
}

bool gpu_sort_int(int* host, int n) {
  if (n <= 1) {
    return true;
  }
  if (!gpu_sort_int_available()) {
    return false;
  }
  if (n < kMinPracticalGpuSortN) {
    return false;
  }

  const size_t bytes = static_cast<size_t>(n) * sizeof(int);

#if defined(PASE_GPU_SORT_USE_CUB)
  std::vector<uint32_t> keys_in(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    keys_in[static_cast<size_t>(i)] =
        static_cast<uint32_t>(host[i]) ^ 0x80000000u;
  }
  uint32_t* d_in = nullptr;
  uint32_t* d_out = nullptr;
  if (!check(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes))) {
    return false;
  }
  if (!check(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes))) {
    cudaFree(d_in);
    return false;
  }
  if (!check(cudaMemcpy(d_in, keys_in.data(), bytes, cudaMemcpyHostToDevice))) {
    cudaFree(d_in);
    cudaFree(d_out);
    return false;
  }

  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_in, d_out, n);
  if (!check(cudaMalloc(&d_temp, temp_bytes))) {
    cudaFree(d_in);
    cudaFree(d_out);
    return false;
  }
  const cudaError_t sort_st = cub::DeviceRadixSort::SortKeys(
      d_temp, temp_bytes, d_in, d_out, n);
  cudaFree(d_temp);
  cudaFree(d_in);
  if (!check(sort_st)) {
    cudaFree(d_out);
    return false;
  }
  if (!check(cudaGetLastError())) {
    cudaFree(d_out);
    return false;
  }

  std::vector<uint32_t> keys_out(static_cast<size_t>(n));
  if (!check(cudaMemcpy(keys_out.data(), d_out, bytes, cudaMemcpyDeviceToHost))) {
    cudaFree(d_out);
    return false;
  }
  cudaFree(d_out);
  for (int i = 0; i < n; ++i) {
    host[i] = static_cast<int>(keys_out[static_cast<size_t>(i)] ^ 0x80000000u);
  }
  return true;
#else
  return gpu_sort_thrust_t(host, n);
#endif
}

}  // namespace pase
