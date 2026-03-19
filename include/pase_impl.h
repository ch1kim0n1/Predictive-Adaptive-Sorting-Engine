#pragma once

#include "cpu_algorithms.h"
#include "dispatcher.h"
#include "pase.h"
#include "profiler.h"

#include <iostream>
#include <iomanip>

namespace pase {

namespace {

[[maybe_unused]] static void print_verbose(const Profile& p, Strategy s) {
  std::cout << "\n[PASE Profiler]\n";
  std::cout << "  n               = " << std::fixed << p.n << "\n";
  std::cout << "  sample_rate     = " << std::setprecision(1)
            << (p.sample_rate * 100) << "%  ("
            << static_cast<int>(p.n * p.sample_rate) << " elements scanned)\n";
  std::cout << "  sortedness      = " << std::setprecision(2) << p.sortedness
            << "\n";
  std::cout << "  avg_run_length  = " << p.avg_run_length << "\n";
  std::cout << "  duplicate_ratio = " << p.duplicate_ratio << "\n";
  std::cout << "  entropy         = " << p.entropy;
  if (p.entropy < 0.3f) {
    std::cout << "  (low — structured data)";
  } else if (p.entropy > 0.7f) {
    std::cout << "  (high — random data)";
  }
  std::cout << "\n\n[PASE Dispatcher]\n";
  std::cout << "  Decision        => ";
  switch (s) {
    case Strategy::INSERTION_OPT:
      std::cout << "INSERTION_OPT  (nearly sorted)\n";
      break;
    case Strategy::RUN_MERGE_OPT:
      std::cout << "RUN_MERGE_OPT  (long runs)\n";
      break;
    case Strategy::THREE_WAY_QS:
      std::cout << "THREE_WAY_QS   (heavy duplicates)\n";
      break;
    case Strategy::INTROSORT:
      std::cout << "INTROSORT      (fallback)\n";
      break;
    case Strategy::GPU_SORT:
      std::cout << "GPU_SORT       (Phase 3)\n";
      break;
  }
}

template <typename T, typename Comp>
void execute_strategy(T* array, int n, Strategy s, const Comp& comp) {
  switch (s) {
    case Strategy::INSERTION_OPT:
      cpu::insertion_sort(array, n, comp);
      break;
    case Strategy::RUN_MERGE_OPT:
      cpu::run_merge_sort(array, n, comp);
      break;
    case Strategy::THREE_WAY_QS:
      cpu::quicksort_3way(array, n, comp);
      break;
    case Strategy::INTROSORT:
    case Strategy::GPU_SORT:
      cpu::introsort(array, n, comp);
      break;
  }
}

}  // namespace

template <typename T, typename Comp>
void adaptive_sort(T* array, int n, const Comp& comp, bool verbose) {
  if (n <= 1) return;

  if (n < 1000) {
    cpu::introsort(array, n, comp);
    return;
  }

  Profiler profiler(0.015f);
  Profile p = profiler.analyze(array, n, comp);

  Dispatcher dispatcher;
  Strategy s = dispatcher.select_strategy(p);

  if (verbose) {
    print_verbose(p, s);
  }

  execute_strategy(array, n, s, comp);
}

template <typename T, typename Comp>
void adaptive_sort(std::vector<T>& vec, const Comp& comp, bool verbose) {
  if (!vec.empty()) {
    adaptive_sort<T, Comp>(vec.data(), static_cast<int>(vec.size()), comp, verbose);
  }
}

}  // namespace pase
