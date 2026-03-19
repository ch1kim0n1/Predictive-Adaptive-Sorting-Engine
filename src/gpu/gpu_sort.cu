#include "gpu_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

namespace pase {

namespace {

__global__ void k_bitonic_step(int* d, int n, int j, int k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int ixj = i ^ j;
  if (ixj <= i || ixj >= n) return;
  const bool up = ((i & k) == 0);
  const int ai = d[i];
  const int aj = d[ixj];
  if (up) {
    if (ai > aj) {
      d[i] = aj;
      d[ixj] = ai;
    }
  } else {
    if (ai < aj) {
      d[i] = aj;
      d[ixj] = ai;
    }
  }
}

int next_pow2(int x) {
  int p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

bool check(cudaError_t e) { return e == cudaSuccess; }

}  // namespace

bool gpu_sort_int_available() {
  int nd = 0;
  if (cudaGetDeviceCount(&nd) != cudaSuccess) {
    return false;
  }
  return nd > 0;
}

bool gpu_sort_int(int* host, int n) {
  if (n <= 1) {
    return true;
  }
  if (!gpu_sort_int_available()) {
    return false;
  }

  const int cap = next_pow2(n);
  const size_t bytes = static_cast<size_t>(cap) * sizeof(int);

  std::vector<int> padded(static_cast<size_t>(cap));
  std::memcpy(padded.data(), host, static_cast<size_t>(n) * sizeof(int));
  std::fill(padded.begin() + n, padded.end(), INT_MAX);

  int* d = nullptr;
  if (!check(cudaMalloc(reinterpret_cast<void**>(&d), bytes))) {
    return false;
  }

  cudaStream_t stream{};
  if (!check(cudaStreamCreate(&stream))) {
    cudaFree(d);
    return false;
  }

  if (!check(cudaMemcpyAsync(d, padded.data(), bytes, cudaMemcpyHostToDevice,
                             stream))) {
    cudaStreamDestroy(stream);
    cudaFree(d);
    return false;
  }

  constexpr int kThreads = 256;
  const int blocks = (cap + kThreads - 1) / kThreads;

  for (int k = 2; k <= cap; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      k_bitonic_step<<<blocks, kThreads, 0, stream>>>(d, cap, j, k);
    }
  }

  if (!check(cudaGetLastError())) {
    cudaStreamDestroy(stream);
    cudaFree(d);
    return false;
  }

  if (!check(cudaMemcpyAsync(host, d, static_cast<size_t>(n) * sizeof(int),
                             cudaMemcpyDeviceToHost, stream))) {
    cudaStreamDestroy(stream);
    cudaFree(d);
    return false;
  }

  if (!check(cudaStreamSynchronize(stream))) {
    cudaStreamDestroy(stream);
    cudaFree(d);
    return false;
  }

  cudaStreamDestroy(stream);
  cudaFree(d);
  return true;
}

}  // namespace pase
