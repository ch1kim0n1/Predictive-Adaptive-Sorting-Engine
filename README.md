# PASE — Predictive Adaptive Sorting Engine

CPU + CUDA Hybrid — Self-Tuning Runtime Architecture

PASE is a systems-level sorting framework that combines runtime data profiling, cost-model-driven dispatch, adaptive CPU algorithm selection, and (Phase 3+) CUDA-accelerated GPU kernels into a single unified pipeline.

## Status

- **Phase 0**: Project setup ✓
- **Phase 1**: Profiler + rule-based dispatcher ✓
- **Phase 2**: Cost model + RUN_MERGE / 3-way QS + feedback CSV EMA-ready ✓
- **Phase 3**: CUDA int sort (optional) + runtime GPU dispatch + online GPU margin tuning ✓

## Building

Requires **CMake 3.17+** (for `find_package(CUDAToolkit)` when CUDA is enabled).

Default (CPU only; GPU stub):

```bash
mkdir build && cd build
cmake ..
make
```

**With NVIDIA CUDA** (Linux / Windows; not macOS):

```bash
cmake .. -DPASE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native   # or e.g. 75, 86 for your GPU
make
```

Defines `PASE_WITH_CUDA` so `gpu_sort_int_available()` can be true at runtime.

For release build with optimizations:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Running Tests

```bash
cd build
ctest -R "Correctness|Profiler|CostModel|GpuSort"
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

## Feedback log (Phase 2–3)

Sort decisions and timings are appended to `~/.pase/sort_log.csv` when either:

- Environment variable `PASE_FEEDBACK=1`, or
- `pase::set_feedback_logging(true)` (see `feedback.h`)

## GPU sort (Phase 3)

- **Algorithm**: host-padded power-of-2 array, global **bitonic sort** on device, `cudaMemcpyAsync` + stream (see `src/gpu/gpu_sort.cu`).
- **Dispatch**: `GPU_SORT` only when `PASE_WITH_CUDA` is built **and** `gpu_sort_int_available()` and cost model + `gpu_win_factor()` favor GPU (`include/threshold_tuner.h`).
- **`adaptive_sort` uses GPU only for `int` with `std::less<int>`**; other types fall back to CPU if `GPU_SORT` is chosen.
- After each GPU sort, `ThresholdTuner` nudges the win factor from predicted vs actual time.

## License

MIT — see [LICENSE](LICENSE)
