# PASE — Predictive Adaptive Sorting Engine

CPU + CUDA Hybrid — Self-Tuning Runtime Architecture

PASE is a systems-level sorting framework that combines runtime data profiling, cost-model-driven dispatch, adaptive CPU algorithm selection, and (Phase 3+) CUDA-accelerated GPU kernels into a single unified pipeline.

## Status

- **Phase 0**: Project setup ✓
- **Phase 1**: Profiler + rule-based dispatcher ✓
- **Phase 2**: Cost model + RUN_MERGE / 3-way QS + feedback CSV EMA-ready ✓
- **Phase 3**: CUDA int sort (optional) + runtime GPU dispatch + online GPU margin tuning ✓
- **Phase 4**: JSON thresholds + `bench_results` CSV + tuning scripts ✓

## Phase 4 — config, benchmark CSV, plots

**Threshold file**: `~/.pase/optimized_thresholds.json` (override with `PASE_CONFIG=/path/to.json`).
Must be **valid JSON** (parsed with [nlohmann/json](https://github.com/nlohmann/json)). Partial keys are OK;
supported keys: `sorted`, `run_merge`, `dup`, `min_gpu`, `gpu_win_factor`,
`max_insertion_n`, `strategy_guardrail`, `gpu_rel_margin`, optional
`dup_border_band`, `run_merge_border`, `conservative_specialist_frac`, and
optional **`cost_fit`**
(see [docs/PERF_TUNING.md](docs/PERF_TUNING.md) and `tune/fit_cost_model.py`).

Defaults favor **no large‑n insertion pathologies** (insertion only for `n <= max_insertion_n`),
**run‑merge when sample shows long runs** (`run_merge` default 32), stricter **GPU pick** (`min_gpu`,
`gpu_rel_margin`), and **model guardrails** vs introsort.

**Benchmark CSV** (grid of dataset × size; mean/stdev for PASE, `std::sort`, `std::stable_sort`):

```bash
cd build
./bench/bench_results --quick --out ../bench_quick.csv
./bench/bench_results --out ../bench_full.csv --sizes 10000,100000,500000 --repeat 9
```

**Tuning helpers** (Python 3):

```bash
cd tune
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 analyze_log.py ../build/bench_quick.csv --col speedup_vs_std
python3 plot_results.py ../build/bench_quick.csv -o speedup.png
python3 grid_search.py ./grid_configs   # emits JSON grid for PASE_CONFIG experiments
python3 score_configs.py ../build/bench_quick.csv   # summarize mean/median speedup_vs_std
python3 fit_cost_model.py ../build/bench_quick.csv   # suggest cost_fit JSON from bench CSV
```

**Evaluation contract**: versioned workloads and acceptance constants are documented in
[docs/BENCHMARK_SUITE.md](docs/BENCHMARK_SUITE.md) (`include/pase_bench_contract.h`).

**Performance acceptance**: `ctest -R PerformanceRegression` checks structured cases and a
random smoke case vs `std::sort` medians using **`pase::bench_contract`** limits.

**CI layout** (CPU vs CUDA jobs): [docs/CI.md](docs/CI.md).

### Limitations (honest scope)

- Wins are **workload-dependent** (see `bench_results`); median vs `std::sort` may be &lt; 1× on mixed/random data.
- **GPU sort** (CUDA) defaults to **Thrust** device sort; optional **CUB** radix path via
  `-DPASE_GPU_SORT_USE_CUB=ON` (see [docs/PERF_TUNING.md](docs/PERF_TUNING.md)).
- **Cost model** defaults to calibration + heuristics; use **`cost_fit`** + `fit_cost_model.py` for offline tuning to your machine (see [docs/PERF_TUNING.md](docs/PERF_TUNING.md)).
- **Profiler** uses sampling; duplicate-rich workloads are estimated better via sorted-sample stats but remain approximate.

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
Optional: add `-DPASE_GPU_SORT_USE_CUB=ON` to compile the experimental CUB radix sorter.

For release build with optimizations:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Running Tests

**PASE tests** (recommended for CI and pre-merge). Run from the **CMake build directory** (the folder that contains `CMakeCache.txt`) — often `…/Predictive-Adaptive-Sorting-Engine/build`. If your shell prompt already ends with `build`, run `ctest` directly (do not run `cd build` again).

```bash
cd /path/to/Predictive-Adaptive-Sorting-Engine/build   # once, from repo root
ctest -R "Correctness|Profiler|CostModel|GpuSort|ConfigLoader|PerformanceRegression" --output-on-failure
```

**Full `ctest`**: By default the project does **not** register Google Benchmark’s
upstream self-tests (they can fail on some platforms, e.g. macOS console output).
A normal configure runs **only PASE tests** when you use `ctest`.

To include upstream Benchmark tests (optional):

```bash
cmake .. -DPASE_BUILD_BENCHMARK_UPSTREAM_TESTS=ON
cmake --build .
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

Columns include **`pred_gpu_transfer_ms`** and **`pred_gpu_kernel_ms`** (PCIe vs
device compute estimates) plus combined predictors; see `src/feedback.cpp`.

## GPU sort (Phase 3)

- **Algorithm**: **Thrust** `thrust::sort` by default on device (`src/gpu/gpu_sort.cu`); optional **CUB** radix build (`-DPASE_GPU_SORT_USE_CUB=ON`).
- **Dispatch**: `GPU_SORT` only when `PASE_WITH_CUDA` is built **and** `gpu_sort_int_available()` and cost model + `gpu_win_factor()` favor GPU (see `include/threshold_tuner.h`, `gpu_rel_margin`, `min_gpu`).
- **`adaptive_sort` uses GPU only for `int` with `std::less<int>`**; other types fall back to CPU if `GPU_SORT` is chosen.
- **`gpu_sort_int` rejects very small `n`** (transfer + launch overhead); see `src/gpu/gpu_sort.cu`.
- After each GPU sort, `ThresholdTuner` nudges the win factor from predicted vs actual time.

## License

MIT — see [LICENSE](LICENSE)
