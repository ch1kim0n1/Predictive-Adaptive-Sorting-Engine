#include <benchmark/benchmark.h>
#include <pase.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "gen_datasets.h"

static void BM_PASE(benchmark::State& state) {
  DatasetType type = static_cast<DatasetType>(state.range(0));
  size_t n = state.range(1);
  std::vector<int> data;
  generate_dataset(data, type, n);
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<int> copy = data;
    state.ResumeTiming();
    pase::adaptive_sort(copy);
  }
  state.SetItemsProcessed(state.iterations() * n);
}

static void BM_StdSort(benchmark::State& state) {
  DatasetType type = static_cast<DatasetType>(state.range(0));
  size_t n = state.range(1);
  std::vector<int> data;
  generate_dataset(data, type, n);
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<int> copy = data;
    state.ResumeTiming();
    std::sort(copy.begin(), copy.end());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  const size_t n = 100000;
  b->Args({static_cast<int64_t>(DatasetType::sorted), n})
      ->Args({static_cast<int64_t>(DatasetType::reverse), n})
      ->Args({static_cast<int64_t>(DatasetType::nearly_sorted_95), n})
      ->Args({static_cast<int64_t>(DatasetType::nearly_sorted_80), n})
      ->Args({static_cast<int64_t>(DatasetType::random), n})
      ->Args({static_cast<int64_t>(DatasetType::heavy_dup), n})
      ->Args({static_cast<int64_t>(DatasetType::clustered), n})
      ->Args({static_cast<int64_t>(DatasetType::long_runs), n})
      ->Args({static_cast<int64_t>(DatasetType::pipe_organ), n});
}

BENCHMARK(BM_PASE)->Apply(CustomArguments)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StdSort)->Apply(CustomArguments)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
