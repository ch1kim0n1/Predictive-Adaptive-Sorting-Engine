#pragma once

#include <functional>

namespace pase {
namespace cpu {

/**
 * INSERTION_OPT: binary search + memmove.
 * For nearly-sorted data (sortedness > 0.90).
 */
template <typename T, typename Comp>
void insertion_sort(T* array, int n, const Comp& comp);

/**
 * INTROSORT: wrapper around std::sort.
 * Fallback for random data.
 */
template <typename T, typename Comp>
void introsort(T* array, int n, const Comp& comp);

/**
 * RUN_MERGE_OPT: run detection + extension + cache-tiled merge.
 * Phase 2 implementation.
 */
template <typename T, typename Comp>
void run_merge_sort(T* array, int n, const Comp& comp);

/**
 * THREE_WAY_QS: fat-partition 3-way quicksort.
 * Phase 2 implementation.
 */
template <typename T, typename Comp>
void quicksort_3way(T* array, int n, const Comp& comp);

}  // namespace cpu
}  // namespace pase

#include "cpu_algorithms_impl.h"
