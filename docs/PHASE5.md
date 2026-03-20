# Phase 5 — Boom update (SIMD, GPU types, ML, MPI)

This document tracks **Phase 5** capabilities beyond the core PASE v1 pipeline.

## 5.1 SIMD profiler (`simd_profiler.cpp`)

- **Int + `std::less<int>`**: pair/run metrics over sampled indices use **AVX2** (x86_64) or **NEON** (AArch64) when `PASE_ENABLE_SIMD_INTRINSICS=ON` (default).
- CMake sets `-mavx2` / MSVC `/arch:AVX2` only for `simd_profiler.cpp`.
- **Scalar fallback** when intrinsics are disabled or on unknown architectures.

## 5.2 Built introsort for `int` (optional)

- `cmake .. -DPASE_ENABLE_BUILT_INTROSORT_INT=ON` switches `cpu::introsort` for `int` to a **median-of-three Hoare** implementation in `cpu/introsort_simd.cpp` (SIMD partition hooks can be added later).
- Default remains **`std::sort`** for maximum parity with libc++.

## 5.3 Multi-type GPU

- **`gpu_sort_float` / `gpu_sort_double`**: Thrust device sort, same **min-n (~8k)** gate as `int`.
- **`gpu_sort_complex_float` / `gpu_sort_complex_double`**: lexicographic `(real, imag)` sort via a POD key + Thrust (host pack/unpack around `std::complex`).
- **`LexicographicComplexLess<T>`** (`lex_complex_order.h`): comparator for `adaptive_sort` to enable the GPU path with CUDA.
- **`gpu_sort_device_available()`**: used for dispatch gating for all numeric GPU paths.
- `adaptive_sort` can choose **GPU_SORT** for `float`/`double` with `std::less<>` when CUDA is enabled.

## 5.4 ML threshold JSON

- Build: `cmake .. -DPASE_ENABLE_ML_TUNING=ON`.
- At runtime, after loading `optimized_thresholds.json`, merges **`PASE_ML_CONFIG`** or **`~/.pase/ml_thresholds.json`** if present (same key shape as dispatcher JSON).
- **Training** (offline):
  - `tune/collect_training_data.py` — merge many bench CSVs.
  - `tune/train_ml_model.py` — heuristic + optional **scikit-learn** RandomForest; writes `ml_thresholds.json`.

## 5.5 Distributed sort (MPI)

- Build: `cmake .. -DPASE_ENABLE_MPI=ON` (requires MPI).
- **`distributed_sort_mpi_int(comm, local_chunk, global_n_hint)`**: **global** sort via rank-0 gather → `adaptive_sort` → balanced **Scatterv**; `local_chunk` is resized per rank. If the gathered payload would exceed **`distributed_sort_mpi_gather_cap_bytes()`** (256 MiB of `int` data), each rank falls back to **local-only** sort (see stderr on rank 0).
- **`distributed_sort_local_int`**: single-node wrapper (always built).
- **ctest**: `MpiPhase5GlobalSort` runs `test_mpi_phase5` under `mpirun -n 2` when MPI is enabled.

## Performance gates (targets)

| Area            | Target |
|-----------------|--------|
| Profiler        | Reduce pair-scan cost on large samples via SIMD |
| CPU introsort   | Optional custom introsort; SIMD partition planned |
| GPU             | Float/double parity with int Thrust path |
| ML              | Optional JSON overrides; RF when sklearn installed |
| MPI             | Rank-0 gather / scatter global sort under byte cap |

## Tests / benches

- `ctest -R Phase5Integration` — SIMD metric reference check, distributed local vs adaptive, float correctness.
- `ctest -R GpuComplexSort` — GPU lexicographic complex (skipped without device).
- `ctest -R MpiPhase5GlobalSort` — MPI global int sort (2 ranks; only with `-DPASE_ENABLE_MPI=ON`).
- `bench_simd_profiler` — Google Benchmark over `Profiler::analyze` (build with default bench target).
