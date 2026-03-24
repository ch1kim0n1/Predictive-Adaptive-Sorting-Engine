#pragma once

#include "cpu_algorithms.h"
#include "feedback.h"
#include "gpu_api.h"
#include "pase.h"
#include "profiler.h"
#include "runtime.h"
#include "threshold_tuner.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <type_traits>

namespace pase {

inline void print_verbose(const Profile& p, Strategy s, double pred_gpu_ms,
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
#ifdef PASE_WITH_CUDA
      std::cout << "GPU_SORT       (CUDA bitonic path)"
                << (gpu_sort_int_available() ? "" : " [fallback: no device]")
                << "\n";
#else
      std::cout << "GPU_SORT       (build without CUDA → std::sort)\n";
#endif
      break;
  }
}

namespace {

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
      cpu::introsort(array, n, comp);
      break;
    case Strategy::GPU_SORT: {
      if constexpr (std::is_same_v<T, int> &&
                    std::is_same_v<Comp, std::less<int>>) {
        if (gpu_sort_int(array, n)) {
          break;
        }
      }
      if constexpr (std::is_same_v<T, float> &&
                    std::is_same_v<Comp, std::less<float>>) {
        if (gpu_sort_float(array, n)) {
          break;
        }
      }
      if constexpr (std::is_same_v<T, double> &&
                    std::is_same_v<Comp, std::less<double>>) {
        if (gpu_sort_double(array, n)) {
          break;
        }
      }
      if constexpr (std::is_same_v<T, std::complex<float>> &&
                    std::is_same_v<Comp, LexicographicComplexLess<float>>) {
        if (gpu_sort_complex_float(array, n)) {
          break;
        }
      }
      if constexpr (std::is_same_v<T, std::complex<double>> &&
                    std::is_same_v<Comp, LexicographicComplexLess<double>>) {
        if (gpu_sort_complex_double(array, n)) {
          break;
        }
      }
      cpu::introsort(array, n, comp);
      break;
    }
  }
}

}  // namespace

template <typename T, typename Comp>
void adaptive_sort(T* array, int n, const Comp& comp, bool verbose) {
  if (n <= 1) return;

  // Fast probe: detect inputs where profiling is unlikely to beat std::sort.
  bool likely_generic = false;
  bool likely_reverse_like = false;
  bool structured_for_specialist = false;
  if (n >= 2) {
    const int probe_n = std::min(n, 2048);
    int desc_edges = 0;
    int eq_edges = 0;
    int runs = 1;
    for (int i = 1; i < probe_n; ++i) {
      const bool less_cur_prev = comp(array[i], array[i - 1]);
      const bool less_prev_cur = comp(array[i - 1], array[i]);
      if (less_cur_prev) {
        ++desc_edges;
        ++runs;
      }
      if (!less_cur_prev && !less_prev_cur) {
        ++eq_edges;
      }
    }
    const double edges = static_cast<double>(std::max(1, probe_n - 1));
    const double desc_ratio = static_cast<double>(desc_edges) / edges;
    const double eq_ratio = static_cast<double>(eq_edges) / edges;
    const double avg_run_probe =
        static_cast<double>(probe_n) / static_cast<double>(std::max(1, runs));

    likely_generic =
        (desc_ratio >= 0.20 && desc_ratio <= 0.80 && eq_ratio < 0.75 &&
         avg_run_probe < 8.0);
    likely_reverse_like =
        (desc_ratio > 0.95 && eq_ratio < 0.10 && avg_run_probe < 4.0);
    structured_for_specialist =
        (desc_ratio < 0.08 && avg_run_probe >= 12.0);
  }

  // Small inputs: only pay profiling if structure likely benefits specialists.
  if (n < 16384 && !structured_for_specialist) {
    std::sort(array, array + n, comp);
    return;
  }

  // Mid-to-large generic/reverse-like inputs: avoid profiling overhead.
  if (n >= 50000 && (likely_generic || likely_reverse_like)) {
    std::sort(array, array + n, comp);
    return;
  }

  CostModel& cm = global_cost_model();
  Profiler profiler(0.015f);
  Profile p = profiler.analyze(array, n, comp);

  const bool gpu_ok = [] {
#ifdef PASE_WITH_CUDA
    return gpu_sort_device_available();
#else
    return false;
#endif
  }();

  const Dispatcher& dispatcher = runtime_dispatcher();
  const double win = global_threshold_tuner().gpu_win_factor();
  Strategy best_cpu =
      cm.best_cpu_strategy(p, dispatcher.thresholds().sorted,
                          dispatcher.thresholds().run_merge,
                          dispatcher.thresholds().dup,
                          dispatcher.thresholds().max_insertion_n);
  Strategy s = dispatcher.select_strategy(p, cm, sizeof(T), gpu_ok, win);

  const double pred_gpu = cm.estimate_gpu(p.n, p.entropy, sizeof(T));
  const double pred_gpu_xfr =
      cm.estimate_gpu_transfer_ms(p.n, sizeof(T));
  const double pred_gpu_kern =
      cm.estimate_gpu_kernel_ms(p.n, p.entropy);
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

  global_threshold_tuner().observe_gpu_decision(pred_gpu, actual_ms, s);

  FeedbackLogger& fl = global_feedback_logger();
  if (fl.enabled()) {
    const double pred_for_strategy =
        (s == Strategy::GPU_SORT) ? cm.estimate_cpu(p, best_cpu)
                                  : cm.estimate_cpu(p, s);
    const bool ok =
        std::abs(actual_ms - pred_for_strategy) <
        0.35 * std::max(actual_ms, 1e-6) + 2.0;
    SortLog log{p.sortedness,
                p.duplicate_ratio,
                p.entropy,
                p.avg_run_length,
                p.n,
                s,
                pred_for_strategy,
                pred_gpu,
                pred_gpu_xfr,
                pred_gpu_kern,
                actual_ms,
                ok};
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
