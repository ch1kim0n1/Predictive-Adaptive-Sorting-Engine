#include <gtest/gtest.h>
#include <gpu_api.h>
#include <lex_complex_order.h>

#include <algorithm>
#include <complex>
#include <random>
#include <vector>

using namespace pase;

namespace {

TEST(GpuComplexSort, StubOrUnavailableWithoutCuda) {
#ifndef PASE_WITH_CUDA
  EXPECT_FALSE(gpu_sort_device_available());
#endif
}

TEST(GpuComplexSort, LexFloatCorrectWhenDevicePresent) {
  if (!gpu_sort_device_available()) {
    GTEST_SKIP() << "No CUDA device or build without PASE_ENABLE_CUDA";
  }

  constexpr int kN = 8192;
  std::vector<std::complex<float>> v(static_cast<std::size_t>(kN));
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> dist(-500.0f, 500.0f);
  for (auto& z : v) {
    z = std::complex<float>(dist(rng), dist(rng));
  }
  std::vector<std::complex<float>> expected = v;
  std::sort(expected.begin(), expected.end(),
            pase::LexicographicComplexLess<float>());

  ASSERT_TRUE(gpu_sort_complex_float(v.data(), kN));
  EXPECT_EQ(v.size(), expected.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    EXPECT_FLOAT_EQ(v[i].real(), expected[i].real()) << " at " << i;
    EXPECT_FLOAT_EQ(v[i].imag(), expected[i].imag()) << " at " << i;
  }
}

TEST(GpuComplexSort, LexDoubleCorrectWhenDevicePresent) {
  if (!gpu_sort_device_available()) {
    GTEST_SKIP() << "No CUDA device or build without PASE_ENABLE_CUDA";
  }

  constexpr int kN = 8192;
  std::vector<std::complex<double>> v(static_cast<std::size_t>(kN));
  std::mt19937 rng(21);
  std::uniform_real_distribution<double> dist(-400.0, 400.0);
  for (auto& z : v) {
    z = std::complex<double>(dist(rng), dist(rng));
  }
  std::vector<std::complex<double>> expected = v;
  std::sort(expected.begin(), expected.end(),
            pase::LexicographicComplexLess<double>());

  ASSERT_TRUE(gpu_sort_complex_double(v.data(), kN));
  EXPECT_EQ(v.size(), expected.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    EXPECT_DOUBLE_EQ(v[i].real(), expected[i].real()) << " at " << i;
    EXPECT_DOUBLE_EQ(v[i].imag(), expected[i].imag()) << " at " << i;
  }
}

}  // namespace
