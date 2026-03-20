#include "introsort_dispatch.h"

#include <algorithm>

namespace pase {
namespace cpu {

#if defined(PASE_SIMD_SORT_BUILT_INTROSORT)

namespace introsort_built {

int floor_log2(int n) {
  int k = 0;
  for (int x = n; x > 1; x >>= 1) {
    ++k;
  }
  return k;
}

void insertion_sort_int(int* a, int n) {
  for (int i = 1; i < n; ++i) {
    int key = a[i];
    int j = i;
    while (j > 0 && a[j - 1] > key) {
      a[j] = a[j - 1];
      --j;
    }
    a[j] = key;
  }
}

inline void median3(int* a, int lo, int hi) {
  int mid = lo + (hi - lo) / 2;
  if (a[mid] < a[lo]) {
    std::swap(a[mid], a[lo]);
  }
  if (a[hi] < a[lo]) {
    std::swap(a[hi], a[lo]);
  }
  if (a[hi] < a[mid]) {
    std::swap(a[hi], a[mid]);
  }
}

int hoare_partition(int* a, int lo, int hi) {
  median3(a, lo, hi);
  const int pivot = a[lo];
  int i = lo - 1;
  int j = hi + 1;
  for (;;) {
    do {
      ++i;
    } while (a[i] < pivot);
    do {
      --j;
    } while (a[j] > pivot);
    if (i >= j) {
      return j;
    }
    std::swap(a[i], a[j]);
  }
}

void introsort_helper(int* a, int lo, int hi, int depth_limit) {
  while (hi - lo > 16) {
    if (depth_limit == 0) {
      std::make_heap(a + lo, a + hi + 1);
      std::sort_heap(a + lo, a + hi + 1);
      return;
    }
    const int p = hoare_partition(a, lo, hi);
    if (p - lo < hi - p) {
      introsort_helper(a, lo, p, depth_limit - 1);
      lo = p + 1;
    } else {
      introsort_helper(a, p + 1, hi, depth_limit - 1);
      hi = p;
    }
    --depth_limit;
  }
  if (hi > lo) {
    insertion_sort_int(a + lo, hi - lo + 1);
  }
}

void run(int* array, int n) {
  const int depth = 2 * std::max(1, floor_log2(n));
  introsort_helper(array, 0, n - 1, depth);
}

}  // namespace introsort_built

#endif

void pase_cpu_introsort_int_dispatch(int* array, int n) {
  if (n <= 1) {
    return;
  }
#if defined(PASE_SIMD_SORT_BUILT_INTROSORT)
  introsort_built::run(array, n);
#else
  std::sort(array, array + n);
#endif
}

}  // namespace cpu
}  // namespace pase
