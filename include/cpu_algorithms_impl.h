#pragma once

#include "cpu_algorithms.h"
#include <algorithm>
#include <cstring>

namespace pase {
namespace cpu {

template <typename T, typename Comp>
void insertion_sort(T* array, int n, const Comp& comp) {
  for (int i = 1; i < n; i++) {
    T key = array[i];
    int lo = 0, hi = i;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (comp(array[mid], key)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    std::memmove(&array[lo + 1], &array[lo], static_cast<size_t>(i - lo) * sizeof(T));
    array[lo] = key;
  }
}

template <typename T, typename Comp>
void introsort(T* array, int n, const Comp& comp) {
  std::sort(array, array + n, comp);
}

template <typename T, typename Comp>
void run_merge_sort(T* array, int n, const Comp& comp) {
  std::sort(array, array + n, comp);
}

template <typename T, typename Comp>
void quicksort_3way(T* array, int n, const Comp& comp) {
  std::sort(array, array + n, comp);
}

}  // namespace cpu
}  // namespace pase
