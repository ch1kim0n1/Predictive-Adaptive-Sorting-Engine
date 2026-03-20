#pragma once

namespace pase {

/**
 * True if CUDA runtime reports at least one device (build must use PASE_ENABLE_CUDA).
 */
bool gpu_sort_int_available();

/**
 * Sort host array of ints on GPU (ascending). Copies H->D, sorts, D->H.
 * When built with CUDA, uses Thrust device sort (radix / merge style), not bitonic.
 * Returns false if CUDA unavailable, n invalid, CUDA error, or n is below an
 * internal minimum (PCIe + kernel lose to CPU for small n).
 * n <= 1 is a no-op and returns true.
 */
bool gpu_sort_int(int* host_data, int n);

}  // namespace pase
