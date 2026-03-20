/**
 * Benchmark harness: CSV with multi-size grids, repeated timings (mean/stdev),
 * baselines std::sort and std::stable_sort.
 *
 * Usage:
 *   bench_results [--quick] [--out path.csv] [--repeat N] [--sizes N1,N2,...]
 */

#include "gen_datasets.h"

#include <pase.h>
#include <pase_plan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_api.h"
#include "pase_bench_contract.h"
#include "profiler.h"
#include "strategies.h"

namespace {

bool flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == name) {
      return true;
    }
  }
  return false;
}

const char* out_path(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--out") {
      return argv[i + 1];
    }
  }
  return "bench_results.csv";
}

int repeat_count(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--repeat") {
      return std::max(1, std::atoi(argv[i + 1]));
    }
  }
  return 7;
}

std::vector<size_t> parse_sizes_default(bool quick) {
  if (quick) {
    return {100000};
  }
  return {10000, 100000, 500000};
}

std::vector<size_t> parse_sizes_arg(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) != "--sizes") {
      continue;
    }
    std::vector<size_t> out;
    std::stringstream ss(argv[i + 1]);
    std::string part;
    while (std::getline(ss, part, ',')) {
      if (part.empty()) {
        continue;
      }
      const unsigned long v = std::stoul(part);
      out.push_back(static_cast<size_t>(v));
    }
    return out;
  }
  return {};
}

struct TimingStats {
  double mean_ms = 0;
  double stdev_ms = 0;
};

TimingStats time_samples(const std::function<void()>& fn, int warmup, int repeats) {
  using clock = std::chrono::steady_clock;
  for (int w = 0; w < warmup; ++w) {
    fn();
  }
  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    auto t0 = clock::now();
    fn();
    auto t1 = clock::now();
    ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
  const double mean = sum / static_cast<double>(ms.size());
  double var = 0;
  for (double x : ms) {
    const double d = x - mean;
    var += d * d;
  }
  if (ms.size() > 1) {
    var /= static_cast<double>(ms.size() - 1);
  } else {
    var = 0;
  }
  return TimingStats{mean, std::sqrt(var)};
}

bool gpu_available() {
#ifdef PASE_WITH_CUDA
  return pase::gpu_sort_int_available();
#else
  return false;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = flag(argc, argv, "--quick");
  const char* csv_path = out_path(argc, argv);
  const int repeats = repeat_count(argc, argv);
  std::vector<size_t> sizes = parse_sizes_arg(argc, argv);
  if (sizes.empty()) {
    sizes = parse_sizes_default(quick);
  }

  std::ofstream out(csv_path);
  if (!out) {
    std::cerr << "bench_results: cannot open " << csv_path << "\n";
    return 1;
  }

  out << "# pase_bench_suite=" << PASE_BENCH_SUITE_VERSION << '\n';
  out << "dataset,n,sortedness,dup_ratio,entropy,avg_run_length,strategy,"
         "pase_ms,pase_stdev_ms,std_ms,std_stdev_ms,stable_ms,stable_stdev_ms,"
         "speedup_vs_std,speedup_vs_stable,pred_gpu_ms,pred_cpu_ms,gpu_"
         "available\n";

  const bool gpu_ok = gpu_available();

  auto run_grid = [&](DatasetType type) {
    for (size_t n_target : sizes) {
      std::vector<int> base;
      generate_dataset(base, type, n_target);
      const size_t n = base.size();

      pase::Profiler prof(0.015f);
      pase::Profile profile = prof.analyze(base.data(), static_cast<int>(n));
      pase::DispatchPreview prev =
          pase::preview_dispatch_for_profile(profile, sizeof(int), gpu_ok);

      std::vector<int> work;

      TimingStats pase_stat = time_samples(
          [&] {
            work = base;
            pase::adaptive_sort(work.data(), static_cast<int>(work.size()));
          },
          1, repeats);

      TimingStats std_stat = time_samples(
          [&] {
            work = base;
            std::sort(work.begin(), work.end());
          },
          1, repeats);

      TimingStats stable_stat = time_samples(
          [&] {
            work = base;
            std::stable_sort(work.begin(), work.end());
          },
          1, repeats);

      const double speed_std =
          std_stat.mean_ms / std::max(pase_stat.mean_ms, 1e-9);
      const double speed_stable =
          stable_stat.mean_ms / std::max(pase_stat.mean_ms, 1e-9);

      out << dataset_name(type) << ',' << n << ',' << profile.sortedness << ','
          << profile.duplicate_ratio << ',' << profile.entropy << ','
          << profile.avg_run_length << ','
          << pase::strategy_name(prev.strategy) << ',' << pase_stat.mean_ms
          << ',' << pase_stat.stdev_ms << ',' << std_stat.mean_ms << ','
          << std_stat.stdev_ms << ',' << stable_stat.mean_ms << ','
          << stable_stat.stdev_ms << ',' << speed_std << ',' << speed_stable
          << ',' << prev.pred_gpu_ms << ',' << prev.pred_cpu_ms << ','
          << (gpu_ok ? 1 : 0) << '\n';
    }
  };

  for (int ti = 0; ti <= static_cast<int>(DatasetType::pipe_organ); ++ti) {
    auto type = static_cast<DatasetType>(ti);
    if (quick && (type == DatasetType::large_random_1M ||
                  type == DatasetType::large_random_10M)) {
      continue;
    }
    run_grid(type);
  }

  if (!quick) {
    for (auto type : {DatasetType::large_random_1M,
                      DatasetType::large_random_10M}) {
      run_grid(type);
    }
  }

  std::cout << "Wrote " << csv_path << " (sizes=" << sizes.size()
            << " x datasets, repeat=" << repeats << ")\n";
  return 0;
}
