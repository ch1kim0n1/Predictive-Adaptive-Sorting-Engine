#include "cost_model.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <vector>

namespace pase {

CostModel::CostModel() : cpu_ops_per_ms_(1e6) {}

void CostModel::apply_fit(const CostModelFit& f) {
  fit_ = f;
}

double CostModel::estimate_gpu_transfer_ms(int n, std::size_t elem_size) const {
  const double bytes =
      static_cast<double>(std::max(0, n)) * static_cast<double>(elem_size);
  return (2.0 * bytes / (kPcieGbPerS * 1e9)) * 1000.0;
}

double CostModel::estimate_gpu_kernel_ms(int n, float entropy) const {
  const double nn = static_cast<double>(std::max(2, n));
  const double lgn = std::log2(nn);
  const double ops = nn * lgn * lgn;
  double kernel_ms = (ops / (kGpuTflops * 1e12)) * 1000.0;
  kernel_ms *= std::max(1e-6, fit_.gpu_kernel_scale);
  double entropy_factor = 1.0 - 0.3 * static_cast<double>(entropy);
  entropy_factor = std::max(0.1, entropy_factor);
  return kernel_ms * entropy_factor;
}

double CostModel::estimate_gpu(int n, float entropy, std::size_t elem_size) const {
  return estimate_gpu_transfer_ms(n, elem_size) +
         estimate_gpu_kernel_ms(n, entropy);
}

double CostModel::estimate_cpu(const Profile& p, Strategy s) const {
  const double n = static_cast<double>(std::max(1, p.n));
  double base = 0.0;
  /* Amortized profiler + dispatch overhead (O(1) dominant term vs cache effects). */
  const double profile_bias =
      (1.0 + 0.02 * (n / 500000.0)) * std::max(1e-6, fit_.profile_bias_mult);
  /* Per-strategy corrections vs introsort-like baseline (tuned conservatively). */
  double strategy_scale = 1.0;
  switch (s) {
    case Strategy::INSERTION_OPT: {
      double disorder =
          std::max(0.0, 1.0 - static_cast<double>(p.sortedness));
      base = n * disorder * n * 0.5;
      break;
    }
    case Strategy::RUN_MERGE_OPT: {
      const int arl = std::max(1, p.avg_run_length);
      const double num_runs = std::max(1.0, n / static_cast<double>(arl));
      base = n * std::log2(std::max(2.0, num_runs));
      break;
    }
    case Strategy::THREE_WAY_QS: {
      double dup = static_cast<double>(p.duplicate_ratio);
      base = n * std::log2(std::max(2.0, n)) * (1.0 - 0.35 * dup);
      break;
    }
    case Strategy::GPU_SORT:
    case Strategy::INTROSORT:
    default:
      base = n * std::log2(std::max(2.0, n));
      break;
  }

  double fit_scale = 1.0;
  switch (s) {
    case Strategy::RUN_MERGE_OPT:
      strategy_scale = 0.88;
      fit_scale = fit_.scale_run_merge;
      break;
    case Strategy::THREE_WAY_QS:
      strategy_scale = 0.95;
      fit_scale = fit_.scale_three_way;
      break;
    case Strategy::INSERTION_OPT:
      strategy_scale = 1.00;
      fit_scale = fit_.scale_insertion;
      break;
    case Strategy::INTROSORT:
      fit_scale = fit_.scale_introsort;
      break;
    default:
      fit_scale = fit_.scale_introsort;
      break;
  }

  fit_scale = std::max(1e-6, fit_scale);
  strategy_scale *= fit_scale;

  const double denom = std::max(1e-300, cpu_ops_per_ms_);
  return (base * profile_bias * strategy_scale) / denom;
}

Strategy CostModel::best_cpu_strategy(const Profile& p,
                                     float sorted_insertion_thr,
                                     int run_merge_thr,
                                     float dup_thr,
                                     int max_insertion_n) const {
  Strategy best = Strategy::INTROSORT;
  double best_cost = estimate_cpu(p, Strategy::INTROSORT);

  if (p.avg_run_length > run_merge_thr) {
    double c = estimate_cpu(p, Strategy::RUN_MERGE_OPT);
    if (c < best_cost) {
      best_cost = c;
      best = Strategy::RUN_MERGE_OPT;
    }
  }
  if (p.duplicate_ratio > dup_thr) {
    double c = estimate_cpu(p, Strategy::THREE_WAY_QS);
    if (c < best_cost) {
      best_cost = c;
      best = Strategy::THREE_WAY_QS;
    }
  }
  if (p.n <= max_insertion_n && p.sortedness > sorted_insertion_thr - 0.08f) {
    double c = estimate_cpu(p, Strategy::INSERTION_OPT);
    if (c < best_cost) {
      best = Strategy::INSERTION_OPT;
    }
  }
  return best;
}

void CostModel::calibrate_with_int_sort(CostModel& out) {
  constexpr int N = 200000;
  std::vector<int> v(static_cast<size_t>(N));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
  for (int& x : v) {
    x = dist(rng);
  }
  auto t0 = std::chrono::steady_clock::now();
  std::sort(v.begin(), v.end());
  auto t1 = std::chrono::steady_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (ms < 1e-6) {
    ms = 1e-6;
  }
  const double ops =
      static_cast<double>(N) * std::log2(static_cast<double>(std::max(2, N)));
  out.set_cpu_ops_per_ms(ops / ms);
}

}  // namespace pase
