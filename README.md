# PASE — Predictive Adaptive Sorting Engine

CPU + CUDA Hybrid — Self-Tuning Runtime Architecture

PASE is a systems-level sorting framework that combines runtime data profiling, cost-model-driven dispatch, adaptive CPU algorithm selection, and (Phase 3+) CUDA-accelerated GPU kernels into a single unified pipeline.

## Status

- **Phase 0**: Project setup ✓
- **Phase 1**: Profiler + Dispatcher ✓

## Building

```bash
mkdir build && cd build
cmake ..
make
```

For release build with optimizations:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Running Tests

```bash
cd build
ctest -R "Correctness|Profiler"
```

Or run all tests:
```bash
ctest --output-on-failure
```

## Running Benchmarks

```bash
cd build
./bench/bench_main --benchmark_filter="BM_PASE|BM_StdSort"
```

To export results to CSV:
```bash
./bench/bench_main --benchmark_out=results.csv --benchmark_out_format=csv
```

## License

MIT — see [LICENSE](LICENSE)
