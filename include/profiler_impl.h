#pragma once

#include "profiler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

namespace pase {
namespace profiler_detail {

/** Pair-level sortedness + duplicate estimate (sorted-sample adjacency max). */
template <typename T, typename Comp>
void pair_and_sorted_dup_stats(const std::vector<T>& samples, const Comp& comp,
                               float& sortedness, float& duplicate_ratio) {
  sortedness = 1.0f;
  duplicate_ratio = 0.0f;
  if (samples.size() < 2) {
    return;
  }
  int in_order = 0;
  int duplicates = 0;
  int total_pairs = 0;
  for (size_t i = 1; i < samples.size(); ++i) {
    const T& prev = samples[i - 1];
    const T& curr = samples[i];
    if (comp(prev, curr)) {
      in_order++;
    } else if (!comp(curr, prev)) {
      duplicates++;
    }
    total_pairs++;
  }
  sortedness =
      total_pairs > 0 ? static_cast<float>(in_order) / total_pairs : 1.0f;
  const float dup_adjacent =
      total_pairs > 0 ? static_cast<float>(duplicates) / total_pairs : 0.0f;

  std::vector<T> sorted = samples;
  std::sort(sorted.begin(), sorted.end(), comp);
  int adj_equal = 0;
  for (size_t i = 1; i < sorted.size(); ++i) {
    if (!comp(sorted[i - 1], sorted[i]) &&
        !comp(sorted[i], sorted[i - 1])) {
      adj_equal++;
    }
  }
  const float dup_from_sorted =
      sorted.size() > 1 ? static_cast<float>(adj_equal) /
                              static_cast<float>(sorted.size() - 1)
                        : 0.0f;
  duplicate_ratio = std::max(dup_adjacent, dup_from_sorted);
}

}  // namespace profiler_detail

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

  std::vector<T> samples;
  samples.reserve(static_cast<size_t>(sample_count) + 2);
  for (int i = 0; i < n; i += stride) {
    samples.push_back(array[i]);
  }
  if (samples.size() < 2) {
    samples.push_back(array[n - 1]);
  }
  /* Second-stage (mid-stride) samples only when stride metrics are ambiguous —
   * e.g. dup signal neither clearly low nor high, or order neither sorted nor
   * random — same budget as before when triggered, cheaper when structure is clear. */
  float pre_sorted = 1.0f;
  float pre_dup = 0.0f;
  profiler_detail::pair_and_sorted_dup_stats(samples, comp, pre_sorted, pre_dup);
  const bool dup_ambiguous = pre_dup > 0.12f && pre_dup < 0.55f;
  const bool order_ambiguous = pre_sorted > 0.22f && pre_sorted < 0.78f;
  const bool need_refine =
      stride >= 2 && static_cast<int>(samples.size()) < 3000 &&
      (dup_ambiguous || order_ambiguous);
  if (need_refine) {
    for (int ii = stride / 2; ii < n; ii += stride) {
      samples.push_back(array[ii]);
    }
  }

  T min_val = samples[0];
  T max_val = samples[0];
  for (const T& v : samples) {
    if (comp(v, min_val)) {
      min_val = v;
    }
    if (comp(max_val, v)) {
      max_val = v;
    }
  }

  int in_order = 0;
  int total_pairs = 0;
  int duplicates = 0;
  int run_sum = 0;
  int run_count = 0;
  int max_run = 1;
  int current_run = 1;

  for (size_t i = 1; i < samples.size(); ++i) {
    const T& prev = samples[i - 1];
    const T& curr = samples[i];

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

  /* Duplicate density from sorted sample: catches non-adjacent duplicates. */
  std::vector<T> sorted_samples = samples;
  std::sort(sorted_samples.begin(), sorted_samples.end(), comp);
  int adj_equal = 0;
  for (size_t i = 1; i < sorted_samples.size(); ++i) {
    if (!comp(sorted_samples[i - 1], sorted_samples[i]) &&
        !comp(sorted_samples[i], sorted_samples[i - 1])) {
      adj_equal++;
    }
  }
  const float dup_from_value_distribution =
      sorted_samples.size() > 1
          ? static_cast<float>(adj_equal) /
                static_cast<float>(sorted_samples.size() - 1)
          : 0.0f;
  p.duplicate_ratio = std::max(p.duplicate_ratio, dup_from_value_distribution);

  int histogram[256];
  std::memset(histogram, 0, sizeof(histogram));
  int actual_samples = static_cast<int>(samples.size());
  for (const T& val : samples) {
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
