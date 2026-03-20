# Continuous integration expectations

## Default: CPU-only

- **Always run** on macOS / Linux agents **without** CUDA: configure with
  `cmake ..` (leave `PASE_ENABLE_CUDA=OFF`, `PASE_ENABLE_MPI=OFF`, defaults).
- **`ctest`** (default configure) runs **PASE tests only**. Google Benchmark’s
  upstream self-tests are **off** unless you pass
  `-DPASE_BUILD_BENCHMARK_UPSTREAM_TESTS=ON` (they can fail on some hosts due to
  unrelated console-format assertions).

### Recommended full CPU pass

```bash
ctest --output-on-failure
```

This includes, among others:

| Test | Notes |
|------|--------|
| `Correctness` | Core sort correctness |
| `Profiler` | Profiler behavior |
| `CostModel` | Model / dispatcher logic |
| `GpuSort` | GPU API tests (skip without CUDA device) |
| `GpuComplexSort` | Complex GPU path (skip without CUDA device) |
| `ConfigLoader` | JSON thresholds |
| `PerformanceRegression` | Median-time bounds vs `std::sort` (`pase::bench_contract`) |
| `Phase5Integration` | Phase 5 smoke (SIMD metrics, local distributed wrapper, float) |

Optional explicit filter:

```bash
ctest -R 'Correctness|Profiler|CostModel|GpuSort|GpuComplexSort|ConfigLoader|PerformanceRegression|Phase5Integration' \
  --output-on-failure
```

Thresholds are documented in `include/pase_bench_contract.h` and [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md) (suite **v1.2**).

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
  require hardware should **skip** when no device is present (must not fail).

## Optional: MPI job

- Requires MPI (e.g. OpenMPI). Configure with `-DPASE_ENABLE_MPI=ON` (see [PHASE5.md](PHASE5.md)).
- Register **`MpiPhase5GlobalSort`**: runs `test_mpi_phase5` under `mpirun -n 2`.

  ```bash
  cmake .. -DPASE_ENABLE_MPI=ON
  cmake --build . -j
  ctest --output-on-failure
  ```

## Artifacts

- For published benchmark CSVs, record OS, CPU model, `CMAKE_BUILD_TYPE`,
  commit hash, and whether CUDA/CUB/MPI were enabled. See
  [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md).

- Pre-tag verification: [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
