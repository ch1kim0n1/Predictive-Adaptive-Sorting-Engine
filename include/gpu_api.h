#pragma once

namespace pase {

/**
 * True if CUDA runtime reports at least one device (build must use PASE_ENABLE_CUDA).
 */
bool gpu_sort_int_available();

/**
 * Sort host array of ints on GPU (ascending). Copies H->D, sorts, D->H.
 * Returns false if CUDA unavailable, n invalid, or CUDA error.
 * n must be >= 2 for device path; n<=1 is no-op true.
 */
bool gpu_sort_int(int* host_data, int n);

}  // namespace pase
