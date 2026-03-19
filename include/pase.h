#pragma once

#include <functional>
#include <vector>

#include "profiler.h"
#include "strategies.h"

namespace pase {

/**
 * Main entry point: adaptive sort.
 * Profiles input, dispatches to optimal strategy, sorts in-place.
 *
 * @param array Pointer to array
 * @param n Number of elements
 * @param comp Comparator (default: std::less<T>)
 * @param verbose If true, print diagnostic output
 */
template <typename T, typename Comp = std::less<T>>
void adaptive_sort(T* array, int n, const Comp& comp = std::less<T>(),
                  bool verbose = false);

/**
 * Vector overload.
 */
template <typename T, typename Comp = std::less<T>>
void adaptive_sort(std::vector<T>& vec, const Comp& comp = std::less<T>(),
                  bool verbose = false);

}  // namespace pase

#include "pase_impl.h"
