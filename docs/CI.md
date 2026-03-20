# Continuous integration expectations

## Default: CPU-only

- **Always run** on macOS / Linux agents **without** CUDA: configure with
  `cmake ..` (leave `PASE_ENABLE_CUDA=OFF`, the default).
- **`ctest`** (default configure) runs **PASE tests only**. Google Benchmark’s
  upstream self-tests are **off** unless you pass
  `-DPASE_BUILD_BENCHMARK_UPSTREAM_TESTS=ON` (they can fail on some hosts due to
  unrelated console-format assertions).
- Use an explicit filter for clarity:

  ```bash
  ctest -R 'Correctness|Profiler|CostModel|GpuSort|ConfigLoader|PerformanceRegression' \
    --output-on-failure
  ```

- Those tests cover correctness, cost-model/dispatcher logic, config JSON
  parsing, profiler behavior, and **PerformanceRegression** (structured + random
  smoke vs `std::sort`). Thresholds are tied to `pase::bench_contract` in
  `include/pase_bench_contract.h`.

## Optional: CUDA job

- On a **labeled** runner with an NVIDIA GPU and toolkit installed:

  ```bash
  cmake .. -DPASE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
  cmake --build . -j
  ctest --output-on-failure
  ```

- **Experimental radix path**: add `-DPASE_GPU_SORT_USE_CUB=ON` to compile the
  CUB `DeviceRadixSort` branch in `src/gpu/gpu_sort.cu` (default remains Thrust).

- GPU availability is **runtime-gated** (`gpu_sort_int_available()`); tests that
  require hardware should skip when no device is present.

## Artifacts

- For published benchmark CSVs, record OS, CPU model, `CMAKE_BUILD_TYPE`,
  commit hash, and whether CUDA/CUB were enabled. See
  [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md).
