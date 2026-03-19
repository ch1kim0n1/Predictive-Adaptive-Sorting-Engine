#include "cost_model.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <vector>

namespace pase {

CostModel::CostModel() : cpu_ops_per_ms_(1e6) {}

double CostModel::estimate_gpu(int n, float entropy, std::size_t elem_size) const {
  const double bytes =
      static_cast<double>(std::max(0, n)) * static_cast<double>(elem_size);
  const double transfer_ms =
      (2.0 * bytes / (kPcieGbPerS * 1e9)) * 1000.0;
  const double nn = static_cast<double>(std::max(2, n));
  const double lgn = std::log2(nn);
  const double ops = nn * lgn * lgn;
  const double kernel_ms = (ops / (kGpuTflops * 1e12)) * 1000.0;
  double entropy_factor = 1.0 - 0.3 * static_cast<double>(entropy);
  entropy_factor = std::max(0.1, entropy_factor);
  return transfer_ms + kernel_ms * entropy_factor;
}

double CostModel::estimate_cpu(const Profile& p, Strategy s) const {
  const double n = static_cast<double>(std::max(1, p.n));
  double base = 0.0;

  switch (s) {
    case Strategy::INSERTION_OPT: {
      double disorder =
          std::max(0.0, 1.0 - static_cast<double>(p.sortedness));
      base = n * disorder * n * 0.5;
      break;
    }
    case Strategy::RUN_MERGE_OPT: {
      double runs =
          std::max(1.0, static_cast<double>(std::max(1, p.avg_run_length)));
      base = n * std::log2(std::max(2.0, n / runs));
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

  const double denom = std::max(1e-300, cpu_ops_per_ms_);
  return base / denom;
}

Strategy CostModel::best_cpu_strategy(const Profile& p,
                                     float sorted_insertion_thr,
                                     int run_merge_thr,
                                     float dup_thr) const {
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
  if (p.sortedness > sorted_insertion_thr - 0.08f &&
      p.sortedness <= sorted_insertion_thr) {
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
