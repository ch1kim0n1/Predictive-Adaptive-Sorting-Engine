#pragma once

#include <complex>

namespace pase {

/**
 * True if CUDA runtime reports at least one device (build must use PASE_ENABLE_CUDA).
 */
bool gpu_sort_int_available();

/** True if any GPU sort path may succeed (same as int availability today). */
bool gpu_sort_device_available();

/**
 * Sort host array of ints on GPU (ascending). Copies H->D, sorts, D->H.
 * When built with CUDA, uses Thrust device sort (radix / merge style), not bitonic.
 * Returns false if CUDA unavailable, n invalid, CUDA error, or n is below an
 * internal minimum (PCIe + kernel lose to CPU for small n).
 * n <= 1 is a no-op and returns true.
 */
bool gpu_sort_int(int* host_data, int n);

/** Thrust sort for float/double when built with CUDA (same min-n gate as int). */
bool gpu_sort_float(float* host_data, int n);
bool gpu_sort_double(double* host_data, int n);

/**
 * Lexicographic sort of complex (real, then imag) via Thrust when CUDA is enabled.
 */
bool gpu_sort_complex_float(std::complex<float>* host_data, int n);
bool gpu_sort_complex_double(std::complex<double>* host_data, int n);

}  // namespace pase
