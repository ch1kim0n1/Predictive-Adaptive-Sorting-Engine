#include <gtest/gtest.h>
#include <gpu_api.h>

#include <algorithm>
#include <random>
#include <vector>

using namespace pase;

TEST(GpuSortTest, StubReportsUnavailableWithoutCudaBuild) {
#ifndef PASE_WITH_CUDA
  EXPECT_FALSE(gpu_sort_int_available());
#endif
}

TEST(GpuSortTest, OptionalCorrectnessWhenDevicePresent) {
  if (!gpu_sort_int_available()) {
    GTEST_SKIP() << "No CUDA device or build without PASE_ENABLE_CUDA";
  }

  std::vector<int> v(65536);
  std::mt19937 rng(123);
  for (int& x : v) {
    x = static_cast<int>(rng());
  }
  std::vector<int> expected = v;
  std::sort(expected.begin(), expected.end());

  ASSERT_TRUE(gpu_sort_int(v.data(), static_cast<int>(v.size())));
  EXPECT_EQ(v, expected);
}
