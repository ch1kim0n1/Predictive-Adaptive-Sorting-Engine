#include <benchmark/benchmark.h>

#include <profiler.h>

#include <random>
#include <vector>

static void BM_ProfilerAnalyzeInt(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  std::vector<int> v(static_cast<std::size_t>(n));
  std::mt19937 rng(19);
  for (int& x : v) {
    x = static_cast<int>(rng());
  }
  pase::Profiler prof(0.015f);
  for (auto _ : state) {
    pase::Profile p = prof.analyze(v.data(), n);
    benchmark::DoNotOptimize(p.entropy);
    benchmark::DoNotOptimize(p.sortedness);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(n));
}

BENCHMARK(BM_ProfilerAnalyzeInt)->Arg(100'000)->Arg(1'000'000);

BENCHMARK_MAIN();
