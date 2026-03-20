#pragma once

#include <vector>

namespace pase {
namespace simd_profiler {

/**
 * True when this TU was compiled with SIMD intrinsics for int sample metrics.
 * (AVX2 on x86_64, NEON on AArch64 when CMake enables them.)
 */
bool int_sample_metrics_available();

/**
 * Compute pair statistics + ascending run metrics over int samples.
 * Matches scalar semantics in profiler_impl.h (strict < for in_order, == for dup).
 */
void int_sample_metrics(const std::vector<int>& samples, int& in_order,
                        int& duplicates, int& total_pairs, int& run_sum,
                        int& run_count, int& max_run);

}  // namespace simd_profiler
}  // namespace pase
