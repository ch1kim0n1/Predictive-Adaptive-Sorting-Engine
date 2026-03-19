#include <gtest/gtest.h>
#include <pase.h>
#include <profiler.h>
#include <algorithm>
#include <random>
#include <vector>

using namespace pase;

namespace {

class ProfilerTest : public ::testing::Test {
 protected:
  Profiler profiler_{0.015f};
};

TEST_F(ProfilerTest, FullySortedArray_HasSortednessNearOne) {
  std::vector<int> arr(10000);
  std::iota(arr.begin(), arr.end(), 0);
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_GT(p.sortedness, 0.99f);
  EXPECT_GT(p.avg_run_length, 100);
}

TEST_F(ProfilerTest, FullyReverseSorted_HasSortednessNearZero) {
  std::vector<int> arr(10000);
  std::iota(arr.rbegin(), arr.rend(), 0);
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_LT(p.sortedness, 0.1f);
}

TEST_F(ProfilerTest, RandomArray_HasSortednessNearHalf) {
  std::vector<int> arr(10000);
  std::mt19937 rng(42);
  std::iota(arr.begin(), arr.end(), 0);
  std::shuffle(arr.begin(), arr.end(), rng);
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_GT(p.sortedness, 0.3f);
  EXPECT_LT(p.sortedness, 0.7f);
}

TEST_F(ProfilerTest, ConstantArray_HasEntropyNearZero) {
  std::vector<int> arr(10000, 42);
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_LT(p.entropy, 0.01f);
}

TEST_F(ProfilerTest, UniformRandom_HasHighEntropy) {
  std::vector<int> arr(10000);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(0, 1000000);
  for (int& v : arr) v = dist(rng);
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_GT(p.entropy, 0.5f);
}

TEST_F(ProfilerTest, AscendingSequence_HasLongRunLength) {
  std::vector<int> arr(10000);
  std::iota(arr.begin(), arr.end(), 0);
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_GT(p.avg_run_length, 100);
  EXPECT_GT(p.max_run_length, 100);
}

TEST_F(ProfilerTest, HeavyDuplicates_HasHigherDuplicateRatioThanRandom) {
  std::vector<int> arr(10000);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 9);
  for (int& v : arr) v = dist(rng);
  Profile p_dup = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  std::uniform_int_distribution<int> dist_wide(0, 1000000);
  for (int& v : arr) v = dist_wide(rng);
  Profile p_rand = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_GT(p_dup.duplicate_ratio, p_rand.duplicate_ratio);
}

TEST_F(ProfilerTest, SmallArray_HandlesGracefully) {
  std::vector<int> arr = {1, 2, 3};
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_EQ(p.n, 3);
  EXPECT_GT(p.sortedness, 0.99f);
}

TEST_F(ProfilerTest, SingleElement_HandlesGracefully) {
  std::vector<int> arr = {42};
  Profile p = profiler_.analyze(arr.data(), static_cast<int>(arr.size()));
  EXPECT_EQ(p.n, 1);
  EXPECT_EQ(p.sortedness, 1.0f);
}

}  // namespace
