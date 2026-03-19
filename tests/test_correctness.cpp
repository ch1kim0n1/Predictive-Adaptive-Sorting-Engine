#include <gtest/gtest.h>
#include <pase.h>
#include <algorithm>
#include <random>
#include <vector>

using namespace pase;

namespace {

void AssertSorted(const std::vector<int>& arr) {
  for (size_t i = 1; i < arr.size(); i++) {
    ASSERT_LE(arr[i - 1], arr[i]) << " at index " << i;
  }
}

void AssertSameElements(std::vector<int> arr, std::vector<int> sorted) {
  std::sort(arr.begin(), arr.end());
  std::sort(sorted.begin(), sorted.end());
  ASSERT_EQ(arr, sorted);
}

TEST(CorrectnessTest, EmptyArray) {
  std::vector<int> arr;
  adaptive_sort(arr);
  EXPECT_TRUE(arr.empty());
}

TEST(CorrectnessTest, SingleElement) {
  std::vector<int> arr = {42};
  adaptive_sort(arr);
  EXPECT_EQ(arr.size(), 1u);
  EXPECT_EQ(arr[0], 42);
}

TEST(CorrectnessTest, SortedArray) {
  std::vector<int> arr(1000);
  std::iota(arr.begin(), arr.end(), 0);
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, ReverseSortedArray) {
  std::vector<int> arr(1000);
  std::iota(arr.rbegin(), arr.rend(), 0);
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, RandomArray) {
  std::vector<int> arr(5000);
  std::mt19937 rng(42);
  std::iota(arr.begin(), arr.end(), 0);
  std::shuffle(arr.begin(), arr.end(), rng);
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, AllDuplicates) {
  std::vector<int> arr(1000, 42);
  adaptive_sort(arr);
  AssertSorted(arr);
  EXPECT_EQ(arr.size(), 1000u);
  for (int v : arr) EXPECT_EQ(v, 42);
}

TEST(CorrectnessTest, NearlySorted95) {
  std::vector<int> arr(2000);
  std::iota(arr.begin(), arr.end(), 0);
  std::mt19937 rng(123);
  for (int i = 0; i < static_cast<int>(arr.size() * 0.05); i++) {
    int a = rng() % arr.size();
    int b = rng() % arr.size();
    std::swap(arr[a], arr[b]);
  }
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, NearlySorted80) {
  std::vector<int> arr(2000);
  std::iota(arr.begin(), arr.end(), 0);
  std::mt19937 rng(456);
  for (int i = 0; i < static_cast<int>(arr.size() * 0.20); i++) {
    int a = rng() % arr.size();
    int b = rng() % arr.size();
    std::swap(arr[a], arr[b]);
  }
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, HeavyDuplicates) {
  std::vector<int> arr(2000);
  std::mt19937 rng(789);
  std::uniform_int_distribution<int> dist(0, 9);
  for (int& v : arr) v = dist(rng);
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, Clustered) {
  std::vector<int> arr(2000);
  std::mt19937 rng(101);
  std::uniform_int_distribution<int> cluster(0, 4);
  std::uniform_int_distribution<int> offset(0, 100);
  for (int& v : arr) v = cluster(rng) * 1000 + offset(rng);
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, PipeOrgan) {
  std::vector<int> arr(2000);
  size_t mid = arr.size() / 2;
  for (size_t i = 0; i < mid; i++) arr[i] = static_cast<int>(i);
  for (size_t i = mid; i < arr.size(); i++) arr[i] = static_cast<int>(arr.size() - i - 1);
  std::vector<int> copy = arr;
  adaptive_sort(arr);
  AssertSorted(arr);
  AssertSameElements(copy, arr);
}

TEST(CorrectnessTest, RawPointer) {
  std::vector<int> arr = {5, 2, 8, 1, 9};
  adaptive_sort(arr.data(), static_cast<int>(arr.size()));
  AssertSorted(arr);
}

TEST(CorrectnessTest, VerboseModeRunsWithoutError) {
  std::vector<int> arr(5000);
  std::iota(arr.begin(), arr.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(arr.begin(), arr.end(), rng);
  adaptive_sort(arr, std::less<int>(), true);
  AssertSorted(arr);
}

}  // namespace
