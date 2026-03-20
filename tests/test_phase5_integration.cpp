#include <distributed_sort.h>
#include <gtest/gtest.h>
#include <pase.h>
#include <simd_profiler.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace {

TEST(Phase5SimdProfiler, IntPairCountsMatchReference) {
  std::vector<int> samples(500);
  std::mt19937 rng(7);
  std::iota(samples.begin(), samples.end(), 0);
  std::shuffle(samples.begin(), samples.end(), rng);

  int io = 0, dup = 0, tp = 0, rs = 0, rc = 0, mr = 1;
  pase::simd_profiler::int_sample_metrics(samples, io, dup, tp, rs, rc, mr);

  int ref_io = 0, ref_dup = 0, ref_tp = 0;
  for (size_t i = 1; i < samples.size(); ++i) {
    const int a = samples[i - 1];
    const int b = samples[i];
    if (a < b) {
      ref_io++;
    } else if (a == b) {
      ref_dup++;
    }
    ref_tp++;
  }
  EXPECT_EQ(io, ref_io);
  EXPECT_EQ(dup, ref_dup);
  EXPECT_EQ(tp, ref_tp);
}

TEST(Phase5Distributed, LocalIntMatchesAdaptiveSort) {
  std::vector<int> a(800);
  std::iota(a.begin(), a.end(), 0);
  std::mt19937 rng(11);
  std::shuffle(a.begin(), a.end(), rng);
  std::vector<int> b = a;
  pase::distributed_sort_local_int(a.data(), static_cast<int>(a.size()));
  pase::adaptive_sort(b.data(), static_cast<int>(b.size()));
  EXPECT_EQ(a, b);
}

TEST(Phase5Adaptive, FloatSortsCorrectly) {
  std::vector<float> v(2000);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<float>((i * 9973) % 1001) * 0.25f;
  }
  std::mt19937 rng(3);
  std::shuffle(v.begin(), v.end(), rng);
  std::vector<float> copy = v;
  pase::adaptive_sort(v.data(), static_cast<int>(v.size()));
  std::sort(copy.begin(), copy.end());
  EXPECT_EQ(v.size(), copy.size());
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_FLOAT_EQ(v[i], copy[i]) << " at " << i;
  }
}

}  // namespace
