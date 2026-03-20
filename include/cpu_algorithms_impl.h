#pragma once

#include "cpu_algorithms.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace pase {
namespace cpu {

namespace detail {

/** TimSort-style minrun (Java/Python reference). */
inline int compute_minrun(int n) {
  int r = 0;
  int nn = n;
  while (nn >= 64) {
    r |= nn & 1;
    nn >>= 1;
  }
  return nn + r;
}

/** Extend a non-decreasing run [start, end) under comp (strict ascending order). */
template <typename T, typename Comp>
int extend_run(T* A, int start, int n, const Comp& comp) {
  int end = start + 1;
  while (end < n && !comp(A[end], A[end - 1])) {
    ++end;
  }
  return end;
}

constexpr int kMinGallop = 7;

/** Merge sorted [lo, mid) and [mid, hi) using scratch; result in A[lo, hi). */
template <typename T, typename Comp>
void merge_ranges(T* a, int lo, int mid, int hi, const Comp& comp,
                  std::vector<T>& scratch) {
  const int len = hi - lo;
  int i = lo;
  int j = mid;
  int k = 0;
  scratch.resize(static_cast<size_t>(len));
  int consec_left = 0;
  int consec_right = 0;
  while (i < mid && j < hi) {
    if (comp(a[j], a[i])) {
      if (consec_right >= kMinGallop) {
        while (j < hi && comp(a[j], a[i])) {
          scratch[static_cast<size_t>(k++)] = std::move(a[j++]);
        }
        consec_right = 0;
      } else {
        scratch[static_cast<size_t>(k++)] = std::move(a[j++]);
        consec_right++;
        consec_left = 0;
      }
    } else {
      if (consec_left >= kMinGallop) {
        while (i < mid && !comp(a[j], a[i])) {
          scratch[static_cast<size_t>(k++)] = std::move(a[i++]);
        }
        consec_left = 0;
      } else {
        scratch[static_cast<size_t>(k++)] = std::move(a[i++]);
        consec_left++;
        consec_right = 0;
      }
    }
  }
  while (i < mid) {
    scratch[static_cast<size_t>(k++)] = std::move(a[i++]);
  }
  while (j < hi) {
    scratch[static_cast<size_t>(k++)] = std::move(a[j++]);
  }
  for (int t = 0; t < len; ++t) {
    a[lo + t] = std::move(scratch[static_cast<size_t>(t)]);
  }
}

constexpr int kRunMergeInsertionCutoff = 40;
constexpr int kThreeWayInsertionCutoff = 16;

}  // namespace detail

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
    std::memmove(&array[lo + 1], &array[lo],
                 static_cast<size_t>(i - lo) * sizeof(T));
    array[lo] = key;
  }
}

template <typename T, typename Comp>
void introsort(T* array, int n, const Comp& comp) {
  std::sort(array, array + n, comp);
}

template <typename T, typename Comp>
void quicksort_3way(T* array, int n, const Comp& comp) {
  if (n <= 1) return;
  if (n <= detail::kThreeWayInsertionCutoff) {
    insertion_sort(array, n, comp);
    return;
  }
  int lo = 0;
  int hi = n - 1;
  int mid = lo + (hi - lo) / 2;
  if (comp(array[mid], array[lo])) std::swap(array[mid], array[lo]);
  if (comp(array[hi], array[lo])) std::swap(array[hi], array[lo]);
  if (comp(array[hi], array[mid])) std::swap(array[hi], array[mid]);
  std::swap(array[lo], array[mid]);

  int lt = lo;
  int gt = hi;
  int i = lo;
  T pivot = array[lo];

  while (i <= gt) {
    if (comp(array[i], pivot)) {
      std::swap(array[lt++], array[i++]);
    } else if (comp(pivot, array[i])) {
      std::swap(array[i], array[gt--]);
    } else {
      ++i;
    }
  }
  quicksort_3way(array, lt - lo, comp);
  quicksort_3way(array + gt + 1, hi - gt, comp);
}

template <typename T, typename Comp>
void run_merge_sort(T* a, int n, const Comp& comp) {
  if (n <= 1) return;
  if (n <= detail::kRunMergeInsertionCutoff) {
    insertion_sort(a, n, comp);
    return;
  }

  std::vector<std::pair<int, int>> runs;
  int pos = 0;
  while (pos < n) {
    int start = pos;
    int end = detail::extend_run(a, start, n, comp);
    runs.emplace_back(start, end);
    pos = end;
  }

  if (runs.size() <= 1) {
    return;
  }

  const int mr = std::max(1, detail::compute_minrun(n));
  const size_t max_runs = static_cast<size_t>(std::max(8, n / mr));
  if (runs.size() > max_runs) {
    std::sort(a, a + n, comp);
    return;
  }

  std::vector<T> scratch(static_cast<size_t>(n));

  while (runs.size() > 1) {
    std::vector<std::pair<int, int>> next;
    next.reserve((runs.size() + 1) / 2);
    for (size_t r = 0; r < runs.size(); ++r) {
      if (r + 1 < runs.size()) {
        int lo = runs[r].first;
        int mid = runs[r].second;
        int hi = runs[r + 1].second;
        detail::merge_ranges(a, lo, mid, hi, comp, scratch);
        next.emplace_back(lo, hi);
        ++r;
      } else {
        next.push_back(runs[r]);
      }
    }
    runs = std::move(next);
  }
}

}  // namespace cpu
}  // namespace pase
