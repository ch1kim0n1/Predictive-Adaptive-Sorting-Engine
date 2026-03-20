#include "simd_profiler.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(PASE_SIMD_AVX2)
#include <immintrin.h>
#elif defined(PASE_SIMD_NEON)
#include <arm_neon.h>
#endif

namespace pase {
namespace simd_profiler {

namespace {

void int_sample_metrics_scalar(const std::vector<int>& samples, int& in_order,
                               int& duplicates, int& total_pairs, int& run_sum,
                               int& run_count, int& max_run) {
  in_order = duplicates = total_pairs = run_sum = run_count = 0;
  max_run = 1;
  if (samples.size() < 2) {
    return;
  }
  int current_run = 1;
  for (size_t i = 1; i < samples.size(); ++i) {
    const int prev = samples[i - 1];
    const int curr = samples[i];
    if (prev < curr) {
      in_order++;
    } else if (prev == curr) {
      duplicates++;
    }
    total_pairs++;
    if (prev < curr || prev == curr) {
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
}

#if defined(PASE_SIMD_AVX2)

void int_sample_metrics_avx2(const std::vector<int>& samples, int& in_order,
                            int& duplicates, int& total_pairs, int& run_sum,
                            int& run_count, int& max_run) {
  const int n_pairs = static_cast<int>(samples.size()) - 1;
  if (n_pairs <= 0) {
    int_sample_metrics_scalar(samples, in_order, duplicates, total_pairs,
                              run_sum, run_count, max_run);
    return;
  }

  std::vector<uint8_t> kinds(static_cast<size_t>(n_pairs));
  int k = 0;
  for (; k + 8 <= n_pairs; k += 8) {
    __m256i left = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(&samples[static_cast<size_t>(k)]));
    __m256i right = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(&samples[static_cast<size_t>(k + 1)]));
    alignas(32) int L[8];
    alignas(32) int R[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(L), left);
    _mm256_store_si256(reinterpret_cast<__m256i*>(R), right);
    for (int lane = 0; lane < 8; ++lane) {
      const int prev = L[lane];
      const int curr = R[lane];
      if (prev == curr) {
        kinds[static_cast<size_t>(k + lane)] = 2;
      } else if (prev < curr) {
        kinds[static_cast<size_t>(k + lane)] = 1;
      } else {
        kinds[static_cast<size_t>(k + lane)] = 0;
      }
    }
  }
  for (; k < n_pairs; ++k) {
    const int prev = samples[static_cast<size_t>(k)];
    const int curr = samples[static_cast<size_t>(k + 1)];
    if (prev == curr) {
      kinds[static_cast<size_t>(k)] = 2;
    } else if (prev < curr) {
      kinds[static_cast<size_t>(k)] = 1;
    } else {
      kinds[static_cast<size_t>(k)] = 0;
    }
  }

  in_order = duplicates = total_pairs = 0;
  for (int i = 0; i < n_pairs; ++i) {
    const auto t = kinds[static_cast<size_t>(i)];
    if (t == 1) {
      in_order++;
    } else if (t == 2) {
      duplicates++;
    }
    total_pairs++;
  }

  run_sum = run_count = 0;
  max_run = 1;
  int current_run = 1;
  for (int i = 0; i < n_pairs; ++i) {
    const auto t = kinds[static_cast<size_t>(i)];
    if (t == 1 || t == 2) {
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
}

#elif defined(PASE_SIMD_NEON)

void int_sample_metrics_neon(const std::vector<int>& samples, int& in_order,
                            int& duplicates, int& total_pairs, int& run_sum,
                            int& run_count, int& max_run) {
  const int n_pairs = static_cast<int>(samples.size()) - 1;
  if (n_pairs <= 0) {
    int_sample_metrics_scalar(samples, in_order, duplicates, total_pairs,
                              run_sum, run_count, max_run);
    return;
  }

  std::vector<uint8_t> kinds(static_cast<size_t>(n_pairs));
  int k = 0;
  for (; k + 4 <= n_pairs; k += 4) {
    int32x4_t left = vld1q_s32(&samples[static_cast<size_t>(k)]);
    int32x4_t right = vld1q_s32(&samples[static_cast<size_t>(k + 1)]);
    alignas(16) int L[4];
    alignas(16) int R[4];
    vst1q_s32(L, left);
    vst1q_s32(R, right);
    for (int lane = 0; lane < 4; ++lane) {
      const int prev = L[lane];
      const int curr = R[lane];
      kinds[static_cast<size_t>(k + lane)] =
          static_cast<uint8_t>(prev == curr ? 2 : (prev < curr ? 1 : 0));
    }
  }
  for (; k < n_pairs; ++k) {
    const int prev = samples[static_cast<size_t>(k)];
    const int curr = samples[static_cast<size_t>(k + 1)];
    kinds[static_cast<size_t>(k)] =
        static_cast<uint8_t>(prev == curr ? 2 : (prev < curr ? 1 : 0));
  }

  in_order = duplicates = total_pairs = 0;
  for (int i = 0; i < n_pairs; ++i) {
    const auto t = kinds[static_cast<size_t>(i)];
    if (t == 1) {
      in_order++;
    } else if (t == 2) {
      duplicates++;
    }
    total_pairs++;
  }

  run_sum = run_count = 0;
  max_run = 1;
  int current_run = 1;
  for (int i = 0; i < n_pairs; ++i) {
    const auto t = kinds[static_cast<size_t>(i)];
    if (t == 1 || t == 2) {
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
}

#endif  // SIMD variant

}  // namespace

bool int_sample_metrics_available() {
#if defined(PASE_SIMD_AVX2) || defined(PASE_SIMD_NEON)
  return true;
#else
  return false;
#endif
}

void int_sample_metrics(const std::vector<int>& samples, int& in_order,
                        int& duplicates, int& total_pairs, int& run_sum,
                        int& run_count, int& max_run) {
#if defined(PASE_SIMD_AVX2)
  int_sample_metrics_avx2(samples, in_order, duplicates, total_pairs, run_sum,
                         run_count, max_run);
#elif defined(PASE_SIMD_NEON)
  int_sample_metrics_neon(samples, in_order, duplicates, total_pairs, run_sum,
                         run_count, max_run);
#else
  int_sample_metrics_scalar(samples, in_order, duplicates, total_pairs, run_sum,
                            run_count, max_run);
#endif
}

}  // namespace simd_profiler
}  // namespace pase
