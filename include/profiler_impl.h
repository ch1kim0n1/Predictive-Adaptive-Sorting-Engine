#pragma once

#include "profiler.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

namespace pase {

template <typename T, typename Comp>
Profile Profiler::analyze(const T* array, int n, const Comp& comp) {
  Profile p{};
  p.n = n;
  p.sample_rate = sample_rate_;

  if (n <= 1) {
    p.sortedness = 1.0f;
    p.duplicate_ratio = 0.0f;
    p.entropy = 0.0f;
    p.avg_run_length = n;
    p.max_run_length = n;
    p.value_spread = 0.0f;
    return p;
  }

  const int sample_count = std::max(1, static_cast<int>(n * sample_rate_));
  const int stride = std::max(1, n / sample_count);

  int in_order = 0;
  int total_pairs = 0;
  int duplicates = 0;
  int run_sum = 0;
  int run_count = 0;
  int max_run = 1;
  int current_run = 1;

  T min_val = array[0];
  T max_val = array[0];

  for (int i = stride; i < n; i += stride) {
    const T prev = array[i - stride];
    const T curr = array[i];

    if (comp(prev, curr)) {
      in_order++;
    } else if (!comp(curr, prev)) {
      duplicates++;
    }
    total_pairs++;

    if (comp(prev, curr) || (!comp(curr, prev) && !comp(prev, curr))) {
      current_run++;
    } else {
      run_sum += current_run;
      run_count++;
      max_run = std::max(max_run, current_run);
      current_run = 1;
    }

    if (comp(curr, min_val)) min_val = curr;
    if (comp(max_val, curr)) max_val = curr;
  }
  run_sum += current_run;
  run_count++;
  max_run = std::max(max_run, current_run);

  p.sortedness =
      total_pairs > 0 ? static_cast<float>(in_order) / total_pairs : 1.0f;
  p.duplicate_ratio =
      total_pairs > 0 ? static_cast<float>(duplicates) / total_pairs : 0.0f;
  p.avg_run_length = run_count > 0 ? run_sum / run_count : 1;
  p.max_run_length = max_run;

  int histogram[256];
  std::memset(histogram, 0, sizeof(histogram));
  int actual_samples = 0;
  for (int i = 0; i < n; i += stride) {
    actual_samples++;
    const T val = array[i];
    int bucket;
    if (comp(min_val, max_val)) {
      double range = static_cast<double>(max_val) - static_cast<double>(min_val);
      if (range > 0) {
        double frac =
            (static_cast<double>(val) - static_cast<double>(min_val)) / range;
        bucket = static_cast<int>(frac * 255.0);
        bucket = std::clamp(bucket, 0, 255);
      } else {
        bucket = 0;
      }
    } else {
      bucket = 0;
    }
    histogram[bucket]++;
  }

  double entropy_val = 0.0;
  for (int h : histogram) {
    if (h > 0) {
      double p_i = static_cast<double>(h) / actual_samples;
      entropy_val -= p_i * std::log2(p_i);
    }
  }
  const double max_entropy = std::log2(256);
  p.entropy =
      static_cast<float>(std::min(1.0, std::max(0.0, entropy_val / max_entropy)));

  if (n > 0 && comp(min_val, max_val)) {
    double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    p.value_spread = static_cast<float>(range / n);
  } else {
    p.value_spread = 0.0f;
  }

  return p;
}

}  // namespace pase
