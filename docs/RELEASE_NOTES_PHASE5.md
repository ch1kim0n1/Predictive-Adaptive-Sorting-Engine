# Phase 5 release notes (engineering summary)

## Highlights

- **SIMD int profiler metrics** (AVX2 / NEON) with scalar fallback.
- **Optional Hoare introsort** for `int` (`-DPASE_ENABLE_BUILT_INTROSORT_INT=ON`).
- **GPU Thrust sort** for `float` and `double` with shared dispatch path.
- **ML JSON overlays** (`-DPASE_ENABLE_ML_TUNING=ON`) + Python collection/training scripts.
- **MPI global `int` sort** (`-DPASE_ENABLE_MPI=ON`) — rank-0 gather, `adaptive_sort`, balanced Scatterv (256 MiB gather cap).
- **GPU lexicographic complex** (`gpu_sort_complex_*`) + `LexicographicComplexLess` for `adaptive_sort`.
- **`bench_simd_profiler`** — micro-benchmark `Profiler::analyze`.
- **`test_gpu_complex`**, **`test_mpi_phase5`** (with `mpirun -n 2`) when CUDA/MPI are enabled.

## Validation

Run `ctest` (including `Phase5Integration`, `GpuComplexSort`, and `MpiPhase5GlobalSort` when MPI is on) plus existing PASE tests. Regenerate CSV benchmarks on your cluster before trusting ML exports on new hardware.

## Docs / release

- Benchmark contract version is **`1.2`** (`PASE_BENCH_SUITE_VERSION`); see [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md) and [PERF_TUNING.md](PERF_TUNING.md).
- Use [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before tagging.
