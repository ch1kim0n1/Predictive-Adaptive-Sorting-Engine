#pragma once

#include <functional>

namespace pase {

/**
 * Profile struct: workload characterization from sampling profiler.
 * All metrics computed from a 1-2% sample for O(n) behavior with near-zero overhead.
 */
struct Profile {
  float sortedness;       ///< 0.0 = random, 1.0 = sorted
  float duplicate_ratio;  ///< fraction of elements that are duplicates
  float entropy;         ///< Shannon entropy normalized to [0, 1]
  int avg_run_length;     ///< average ascending run length in sample
  int max_run_length;     ///< longest ascending run in sample
  float value_spread;     ///< (max - min) / n, normalized range
  int n;                  ///< full array size
  float sample_rate;      ///< fraction actually sampled
};

/**
 * Profiler: single-pass sampling profiler.
 * Samples configurable fraction (default 1.5%) and computes six metrics.
 */
class Profiler {
 public:
  explicit Profiler(float sample_rate = 0.015f);

  /**
   * Analyze array and return Profile.
   * Uses evenly-spaced stride sampling for unbiased estimates.
   */
  template <typename T, typename Comp = std::less<T>>
  Profile analyze(const T* array, int n, const Comp& comp = std::less<T>());

 private:
  float sample_rate_;
};

}  // namespace pase

#include "profiler_impl.h"
