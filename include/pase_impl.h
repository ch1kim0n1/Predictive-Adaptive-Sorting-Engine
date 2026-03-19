#pragma once

#include "cost_model.h"
#include "cpu_algorithms.h"
#include "dispatcher.h"
#include "feedback.h"
#include "pase.h"
#include "profiler.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace pase {

namespace {

CostModel& global_cost_model() {
  static CostModel model;
  static std::once_flag once;
  std::call_once(once, [] { CostModel::calibrate_with_int_sort(model); });
  return model;
}

void print_verbose(const Profile& p, Strategy s, double pred_gpu_ms,
                  double pred_cpu_ms) {
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
  std::cout << "\n\n[PASE Cost Model]\n";
  std::cout << std::setprecision(3);
  std::cout << "  GPU est.        = Pred " << pred_gpu_ms << " ms  (PCIe + kernel)\n";
  std::cout << "  CPU est.        = Pred " << pred_cpu_ms << " ms\n";

  std::cout << "\n[PASE Dispatcher]\n";
  std::cout << "  Decision        => ";
  switch (s) {
    case Strategy::INSERTION_OPT:
      std::cout << "INSERTION_OPT  (nearly sorted)\n";
      break;
    case Strategy::RUN_MERGE_OPT:
      std::cout << "RUN_MERGE_OPT  (long runs / structured)\n";
      break;
    case Strategy::THREE_WAY_QS:
      std::cout << "THREE_WAY_QS   (heavy duplicates)\n";
      break;
    case Strategy::INTROSORT:
      std::cout << "INTROSORT      (fallback)\n";
      break;
    case Strategy::GPU_SORT:
      std::cout << "GPU_SORT       (cost model; CPU path until Phase 3)\n";
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

  CostModel& cm = global_cost_model();
  Profiler profiler(0.015f);
  Profile p = profiler.analyze(array, n, comp);

  Dispatcher dispatcher;
  Strategy best_cpu =
      cm.best_cpu_strategy(p, dispatcher.thresholds().sorted,
                          dispatcher.thresholds().run_merge,
                          dispatcher.thresholds().dup);
  Strategy s = dispatcher.select_strategy(p, cm, sizeof(T));

  const double pred_gpu = cm.estimate_gpu(p.n, p.entropy, sizeof(T));
  double pred_cpu_for_log = cm.estimate_cpu(p, s);
  if (s == Strategy::GPU_SORT) {
    pred_cpu_for_log = cm.estimate_cpu(p, best_cpu);
  }

  if (verbose) {
    print_verbose(p, s, pred_gpu, pred_cpu_for_log);
  }

  auto t0 = std::chrono::steady_clock::now();
  execute_strategy(array, n, s, comp);
  auto t1 = std::chrono::steady_clock::now();
  double actual_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  FeedbackLogger& fl = global_feedback_logger();
  if (fl.enabled()) {
    const double pred_for_strategy =
        (s == Strategy::GPU_SORT) ? cm.estimate_cpu(p, best_cpu)
                                  : cm.estimate_cpu(p, s);
    const bool ok =
        std::abs(actual_ms - pred_for_strategy) <
        0.35 * std::max(actual_ms, 1e-6) + 2.0;
    SortLog log{p.sortedness,       p.duplicate_ratio,
                p.entropy,          p.avg_run_length,
                p.n,                s,
                pred_for_strategy,  pred_gpu,
                actual_ms,          ok};
    fl.log(log);
  }
}

template <typename T, typename Comp>
void adaptive_sort(std::vector<T>& vec, const Comp& comp, bool verbose) {
  if (!vec.empty()) {
    adaptive_sort<T, Comp>(vec.data(), static_cast<int>(vec.size()), comp, verbose);
  }
}

}  // namespace pase
