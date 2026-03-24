/**
 * PASE Benchmark Suite - Comprehensive Performance Analysis
 * 
 * Usage:
 *   ./bench_main [--quick] [--csv output.csv] [--help]
 * 
 * Modes:
 *   Default (Full):  Tests multiple sizes (10K, 100K, 500K) across all datasets
 *   --quick:         Fast mode, single size (100K) for quick validation
 * 
 * Output:
 *   Console: Human-readable comparison table
 *   CSV:     Detailed results with statistics (if --csv specified)
 */

#include <pase.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "gen_datasets.h"

// Colors for terminal output
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_RESET "\033[0m"

struct TimingStats {
  double mean_ms = 0;
  double median_ms = 0;
  double stdev_ms = 0;
  double min_ms = 0;
  double max_ms = 0;
};

struct BenchmarkResult {
  std::string dataset_name;
  size_t size;
  TimingStats pase;
  TimingStats std_sort;
  TimingStats stable_sort;
  double speedup_vs_std;
  double speedup_vs_stable;
  int iterations;
};

// Helper functions
bool get_flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == flag) {
      return true;
    }
  }
  return false;
}

std::string get_option(int argc, char** argv, const std::string& option,
                       const std::string& default_val) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == option) {
      return argv[i + 1];
    }
  }
  return default_val;
}

TimingStats measure_sort(const std::function<void()>& sort_fn, int warmup_runs,
                         int bench_runs) {
  using clock = std::chrono::steady_clock;
  
  // Warmup
  for (int i = 0; i < warmup_runs; ++i) {
    sort_fn();
  }
  
  // Benchmark
  std::vector<double> times_ms;
  times_ms.reserve(bench_runs);
  
  for (int i = 0; i < bench_runs; ++i) {
    auto t0 = clock::now();
    sort_fn();
    auto t1 = clock::now();
    times_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  
  // Compute statistics
  std::sort(times_ms.begin(), times_ms.end());
  double sum = 0.0;
  for (double t : times_ms) {
    sum += t;
  }
  double mean = sum / times_ms.size();
  
  double variance = 0.0;
  for (double t : times_ms) {
    double diff = t - mean;
    variance += diff * diff;
  }
  variance /= times_ms.size();
  
  return TimingStats{mean, times_ms[times_ms.size() / 2],
                     std::sqrt(variance), times_ms.front(), times_ms.back()};
}

void print_header() {
  std::cout << "\n" << COLOR_BLUE << "=== PASE Benchmark Suite ===" << COLOR_RESET
            << "\n\n";
}

void print_table_header() {
  std::cout << std::left << std::setw(18) << "Dataset" << std::setw(10) << "Size"
            << std::setw(12) << "PASE (ms)" << std::setw(12) << "Std (ms)"
            << std::setw(14) << "Stable (ms)" << std::setw(12) << "vs Std"
            << std::setw(14) << "vs Stable\n";
  std::cout << std::string(92, '-') << "\n";
}

void print_result(const BenchmarkResult& result) {
  std::string speedup_std =
      COLOR_GREEN + std::to_string(result.speedup_vs_std).substr(0, 5) + "x" + COLOR_RESET;
  std::string speedup_stable = COLOR_GREEN +
                               std::to_string(result.speedup_vs_stable).substr(0, 5) +
                               "x" + COLOR_RESET;
  
  if (result.speedup_vs_std < 1.0) {
    speedup_std =
        COLOR_RED + std::to_string(result.speedup_vs_std).substr(0, 5) + "x" + COLOR_RESET;
  }
  if (result.speedup_vs_stable < 1.0) {
    speedup_stable = COLOR_RED +
                     std::to_string(result.speedup_vs_stable).substr(0, 5) + "x" +
                     COLOR_RESET;
  }
  
  std::ostringstream size_str;
  if (result.size >= 1000000) {
    size_str << (result.size / 1000000) << "M";
  } else if (result.size >= 1000) {
    size_str << (result.size / 1000) << "K";
  } else {
    size_str << result.size;
  }
  
  std::cout << std::left << std::setw(18) << result.dataset_name
            << std::setw(10) << size_str.str() << std::fixed
            << std::setprecision(2) << std::setw(12) << result.pase.mean_ms
            << std::setw(12) << result.std_sort.mean_ms << std::setw(14)
            << result.stable_sort.mean_ms << std::setw(12) << speedup_std
            << speedup_stable << "\n";
}

void write_csv(const std::vector<BenchmarkResult>& results,
               const std::string& csv_path) {
  std::ofstream out(csv_path);
  if (!out) {
    std::cerr << "ERROR: Cannot write to " << csv_path << "\n";
    return;
  }
  
  out << "dataset,size,pase_mean_ms,pase_median_ms,pase_stdev_ms,pase_min_ms,"
         "pase_max_ms,std_mean_ms,std_median_ms,std_stdev_ms,std_min_ms,"
         "std_max_ms,stable_mean_ms,stable_median_ms,stable_stdev_ms,"
         "stable_min_ms,stable_max_ms,speedup_vs_std,speedup_vs_stable,"
         "iterations\n";
  
  for (const auto& r : results) {
    out << r.dataset_name << "," << r.size << "," << r.pase.mean_ms << ","
        << r.pase.median_ms << "," << r.pase.stdev_ms << "," << r.pase.min_ms
        << "," << r.pase.max_ms << "," << r.std_sort.mean_ms << ","
        << r.std_sort.median_ms << "," << r.std_sort.stdev_ms << ","
        << r.std_sort.min_ms << "," << r.std_sort.max_ms << ","
        << r.stable_sort.mean_ms << "," << r.stable_sort.median_ms << ","
        << r.stable_sort.stdev_ms << "," << r.stable_sort.min_ms << ","
        << r.stable_sort.max_ms << "," << r.speedup_vs_std << ","
        << r.speedup_vs_stable << "," << r.iterations << "\n";
  }
  
  std::cout << "CSV output written to: " << csv_path << "\n";
}

int main(int argc, char** argv) {
  bool quick_mode = get_flag(argc, argv, "--quick");
  bool help = get_flag(argc, argv, "--help");
  std::string csv_output = get_option(argc, argv, "--csv", "");
  
  if (help) {
    std::cout << "PASE Benchmark Suite\n"
              << "\nUsage: " << argv[0] << " [options]\n"
              << "\nOptions:\n"
              << "  --quick        Run quick mode (single size 100K)\n"
              << "  --full         Run full mode (default, multiple sizes)\n"
              << "  --csv FILE     Write detailed CSV results to FILE\n"
              << "  --help         Show this help message\n";
    return 0;
  }
  
  print_header();
  
  // Select sizes based on mode
  std::vector<size_t> sizes = quick_mode ? std::vector<size_t>{100000}
                                         : std::vector<size_t>{10000, 100000, 500000};
  
  // Runs per benchmark
  int warmup_runs = 2;
  int bench_runs = 5;
  
  std::cout << "Mode: " << (quick_mode ? "QUICK" : "FULL") << " | "
            << "Sizes: ";
  for (size_t s : sizes) {
    std::cout << (s / 1000) << "K ";
  }
  std::cout << "| Repeats: " << bench_runs << "\n\n";
  
  std::vector<BenchmarkResult> all_results;
  
  // Iterate through dataset types
  std::vector<DatasetType> dataset_types = {
      DatasetType::sorted,      DatasetType::reverse,
      DatasetType::nearly_sorted_95, DatasetType::nearly_sorted_80,
      DatasetType::random,      DatasetType::heavy_dup,
      DatasetType::clustered,   DatasetType::long_runs,
      DatasetType::pipe_organ};
  
  for (const auto& dtype : dataset_types) {
    print_table_header();
    
    for (size_t size : sizes) {
      std::vector<int> data;
      generate_dataset(data, dtype, size);
      
      std::cout << "Testing " << dataset_name(dtype) << " (n=" << size << ")...";
      std::flush(std::cout);
      
      // Measure PASE
      TimingStats pase_time = measure_sort(
          [&data]() { pase::adaptive_sort(data.data(), (int)data.size()); },
          warmup_runs, bench_runs);
      
      // Measure std::sort
      TimingStats std_time = measure_sort(
          [&data]() { std::sort(data.begin(), data.end()); }, warmup_runs,
          bench_runs);
      
      // Measure std::stable_sort
      TimingStats stable_time = measure_sort(
          [&data]() { std::stable_sort(data.begin(), data.end()); }, warmup_runs,
          bench_runs);
      
      double speedup_vs_std = std_time.mean_ms / std::max(pase_time.mean_ms, 1e-9);
      double speedup_vs_stable =
          stable_time.mean_ms / std::max(pase_time.mean_ms, 1e-9);
      
      BenchmarkResult result{dataset_name(dtype), size, pase_time, std_time,
                             stable_time, speedup_vs_std, speedup_vs_stable,
                             bench_runs};
      
      print_result(result);
      all_results.push_back(result);
      
      std::cout << " done\n";
    }
    std::cout << "\n";
  }
  
  if (!csv_output.empty()) {
    write_csv(all_results, csv_output);
  }
  
  return 0;
}
