#include <pase.h>
#include <pase_bench_contract.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

using clock = std::chrono::steady_clock;

double median_ms(const std::function<void()>& fn, int warmup, int samples) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    auto t0 = clock::now();
    fn();
    auto t1 = clock::now();
    ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

/** PASE must not be dramatically slower than std::sort on structured inputs. */
void expect_no_large_regression(const std::vector<int>& base,
                               double max_slowdown_vs_std) {
  std::vector<int> work;
  const double pase = median_ms(
      [&] {
        work = base;
        pase::adaptive_sort(work.data(), static_cast<int>(work.size()));
      },
      2, 15);
  const double stdt = median_ms(
      [&] {
        work = base;
        std::sort(work.begin(), work.end());
      },
      2, 15);
  EXPECT_LE(pase, stdt * max_slowdown_vs_std)
      << "pase_ms=" << pase << " std_ms=" << stdt;
}

}  // namespace

TEST(PerformanceRegression, Sorted100kNotPathological) {
  constexpr int n = 100000;
  std::vector<int> base(static_cast<size_t>(n));
  std::iota(base.begin(), base.end(), 0);
  expect_no_large_regression(
      base, pase::bench_contract::kAcceptFullySortedMaxSlowdown);
}

TEST(PerformanceRegression, NearlySorted95_100kNotPathological) {
  constexpr int n = 100000;
  std::vector<int> base(static_cast<size_t>(n));
  std::iota(base.begin(), base.end(), 0);
  std::mt19937 rng(42);
  const int swaps = n / 20;
  for (int i = 0; i < swaps; ++i) {
    const int a = static_cast<int>(rng() % static_cast<unsigned>(n));
    const int b = static_cast<int>(rng() % static_cast<unsigned>(n));
    std::swap(base[static_cast<size_t>(a)], base[static_cast<size_t>(b)]);
  }
  expect_no_large_regression(
      base, pase::bench_contract::kAcceptStructuredMaxSlowdown);
}

TEST(PerformanceRegression, Random50kWithinLooseContract) {
  constexpr int n = 50000;
  std::vector<int> base(static_cast<size_t>(n));
  std::mt19937 rng(99);
  std::iota(base.begin(), base.end(), 0);
  std::shuffle(base.begin(), base.end(), rng);
  expect_no_large_regression(
      base, pase::bench_contract::kAcceptRandomMaxSlowdown);
}
