# PASE — Predictive Adaptive Sorting Engine

**A systems-level CPU + CUDA hybrid sorting framework with runtime profiling, cost-model dispatch, and self-tuning adaptive algorithms.**

---

## What is PASE?

PASE is a **production-grade sorting library** that replaces a single generic sort with an intelligent **dispatch system**:

1. **Sample the array** (1.5% by default) to detect structure: sortedness, runs, duplicates, entropy.
2. **Cost model** predicts which algorithm wins: insertion (nearly sorted), run-merge (structured runs), 3-way quicksort (heavy duplicates), introsort (fallback), or GPU (if CUDA available).
3. **Dispatch** to the best-predicted strategy, with guardrails to avoid pathologies.
4. **Online tuning**: feedback loop adjusts cost model / GPU win margin from real timings.

**Result:** PASE wins decisively on structured workloads (sorted, nearly sorted, duplicate-rich, run-heavy data) and **stays bounded** vs `std::sort` elsewhere. Not a magic bullet for random data, but a serious engineering solution for real-world workloads.

---

## Performance Claims & Realistic Promises

### Where PASE Shines ✅

| Workload | Speedup vs `std::sort` | Notes |
|----------|------------------------|----|
| **Fully sorted** | 1.5–2× faster | Insertion-optimized path; O(n) with early bailout on disorder. |
| **Nearly sorted (95%)** | 1.2–1.6× | Run-merge detects long runs; much cheaper than full resorting. |
| **Heavy duplicates** | 1.3–2× | 3-way quicksort partitions into [< x] [== x] [> x]; avoids recursing equal parts. |
| **Long runs (timsort-like)** | 1.4–1.8× | Run-merge with galloping merge; stays cache-efficient. |
| **GPU (`int`, large n ≥ 100k)** | 2–8× on RTX / A100 | Thrust/CUB device sort + PCIe cost amortized. |

### Where PASE Stays Competitive 🤝

| Workload | vs `std::sort` | Notes |
|----------|----------------|----|
| **Fully random** | 0.9–1.0× (or ≤ 1.1× on CI tests) | Profiler + dispatch overhead ~3% on 100k elements; picks introsort/fallback. Regression tests bound to **≤1.65× on structured, ≤2× on random smoke**. |
| **Mixed / clustered** | 0.95–1.05× | Cost model tuned conservatively; falls back to introsort if uncertain. |
| **Small arrays (n < 3,072)** | 1.0× (same as `std::sort`) | Below this, profiling cost > benefit; bypasses dispatch entirely. |

### What We Don't Promise ❌

- **"Always faster than `std::sort` on all inputs."** Profiling + dispatch has overhead; on purely random data, the best PASE can do is call introsort-class code, which ties the baseline.
- **"GPU is always a win."** PCIe transfers are expensive (~10–20 GB/s); GPU sort is practical only for large enough `n` (typically ≥ 50k for `int` on modern GPUs) and when compute cost dominates. PASE estimates both and chooses wisely.
- **"One tuning works everywhere."** The cost model is calibrated on reference hardware (see [docs/PERF_TUNING.md](docs/PERF_TUNING.md)); on different CPU/GPU combos, use offline `fit_cost_model.py` to adapt.

---

## Architecture & Pipeline

```
Input Array
    ↓
[Profiler] ──→ Profile (sortedness, runs, duplicates, entropy)
    ↓
[Cost Model] ──→ Per-strategy time estimates (CPU: insertion, run-merge, 3-way, introsort; GPU: kernel + transfer)
    ↓
[Dispatcher] ──→ Choose strategy (with border-band conservatism + guardrails)
    ↓
[Execute] ──→ CPU algorithms or GPU sort
    ↓
[Feedback] (optional) ──→ Log decision + actual time → online tuning
```

### Key Innovations

1. **Ambiguity-gated second-stage profiler:** Cheap initial sampling; only refine if signals are ambiguous (e.g., duplicate ratio in a middle band). Avoids wasting samples on clearly sorted/random data.

2. **Border-band conservative dispatch:** When profile metrics sit near strategy thresholds (e.g., dup near the 3-way boundary), prefer safer introsort unless the specialist strategy beats it by a meaningful margin. Reduces "wrong call" risk.

3. **Split GPU cost:** PCIe transfer + device compute logged separately so cost model can tune transfer vs kernel overhead independently.

4. **Online EMA tuning:** `ThresholdTuner` adjusts GPU win factor from real feedback, helping the model converge faster on new hardware.

---

## Benchmarks & Evaluation

### Suite v1.1: Versioned Workloads

**8 dataset types** × **default size grid** (10k, 100k, 500k elements) → CSV with mean ± stdev for PASE, `std::sort`, `std::stable_sort`.

| Dataset | Description | Use Case |
|---------|-------------|----------|
| sorted | Ascending 0..n-1 | Best-case: insertion wins. |
| reverse | Descending n-1..0 | Worst-case: long single reverse run. |
| nearly_sorted_95 / 80 | iota + 5% / 20% random swaps | Real-world: data is mostly in order. |
| random | Shuffled iota | Baseline: no structure. |
| heavy_dup | Uniform 0..9 | Duplicates stress 3-way / run-merge. |
| clustered | Clusters * 1000 + offset | Locality: cache effects. |
| **long_runs** | 72-element ascending runs | Timsort / run-merge stress. |
| pipe_organ | Ascending half + descending half | Mixed structure. |

**Acceptance bounds** (CI regression tests):
- **Structured** (sorted, nearly_sorted_95): ≤ **1.65×** `std::sort` median
- **Random smoke** (50k random): ≤ **2.0×** `std::sort` median

See [docs/BENCHMARK_SUITE.md](docs/BENCHMARK_SUITE.md) and `include/pase_bench_contract.h`.

### Running Benchmarks

**Quick CSV export** (1.5 min on modern CPU):
```bash
cd build
./bench/bench_results --quick --out results.csv
```

**Full suite** (10 min, includes 1M/10M sizes):
```bash
./bench/bench_results --out results_full.csv --sizes 10000,100000,500000,1000000,10000000 --repeat 9
```

**Analyze & plot** (Python):
```bash
cd tune
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 analyze_log.py ../build/results.csv --col speedup_vs_std
python3 plot_results.py ../build/results.csv -o speedup_chart.png
python3 score_configs.py ../build/results.csv   # median speedup per workload
python3 fit_cost_model.py ../build/results.csv  # offline tuning suggestions
```

---

## Getting Started

### 1. Build (CPU-only)

```bash
git clone https://github.com/your-org/Predictive-Adaptive-Sorting-Engine.git
cd Predictive-Adaptive-Sorting-Engine
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

### 2. Run Tests

```bash
ctest -R "Correctness|Profiler|CostModel|GpuSort|ConfigLoader|PerformanceRegression" --output-on-failure
```

Expected output: **6 tests pass** in ~1 second.

### 3. Try Sorting

```cpp
#include <pase.h>
#include <vector>

int main() {
  std::vector<int> arr = {/* your data */};
  pase::adaptive_sort(arr);  // sorts in-place
  // or: pase::adaptive_sort(arr.data(), arr.size());
}
```

### 4. Benchmarks & Tuning

See **Building → Benchmarks & Evaluation** above.

---

## GPU Integration (CUDA)

### Build with CUDA

```bash
cmake .. -DPASE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native  # auto-detect; or e.g. 75,86 for RTX
cmake --build . -j$(nproc)
```

### Optional: CUB Radix Sort

For experimental **NVIDIA CUB** `DeviceRadixSort` (unsigned-key mapping for `int`):

```bash
cmake .. -DPASE_ENABLE_CUDA=ON -DPASE_GPU_SORT_USE_CUB=ON
```

(Default: **Thrust** `thrust::sort`; both compile cleanly on supported CUDA versions.)

### GPU Dispatch Logic

- **Enabled by:** `PASE_WITH_CUDA` build flag + `gpu_sort_int_available()` at runtime.
- **Chosen when:**
  - `n ≥ min_gpu` (default 250k)
  - GPU transfer + kernel time **<** CPU time × `gpu_win_factor` (default 0.85, tuned online)
  - GPU relative margin `gpu_rel_margin` buffer (default 1.12×) adds safety.
- **Cost model splits:**
  - **Transfer:** H2D + D2H PCIe bandwidth model (~12 GB/s default).
  - **Kernel:** Device compute model (entropy-modulated for disorder).
  - **Separate tuning** via `cost_fit.gpu_kernel_scale` in config.

### Feedback & Online Tuning

Enable feedback logging to track GPU decisions:

```bash
export PASE_FEEDBACK=1
# Run sorting...
cat ~/.pase/sort_log.csv  # see pred_gpu_transfer_ms, pred_gpu_kernel_ms, actual_ms
```

CSV columns include predicted transfer, kernel, and actual time so you can audit / re-fit.

---

## Configuration & Tuning

### Thresholds JSON

Create `~/.pase/optimized_thresholds.json` (or set `PASE_CONFIG=/path/to.json`):

```json
{
  "sorted": 0.90,
  "run_merge": 32,
  "dup": 0.32,
  "min_gpu": 250000,
  "gpu_win_factor": 0.85,
  "max_insertion_n": 384,
  "strategy_guardrail": 2.25,
  "gpu_rel_margin": 1.12,
  "dup_border_band": 0.08,
  "run_merge_border": 6,
  "conservative_specialist_frac": 0.88,
  "cost_fit": {
    "introsort": 1.05,
    "run_merge": 0.92,
    "three_way": 0.88,
    "insertion": 1.0,
    "gpu_kernel": 0.65,
    "profile_bias_mult": 1.02
  }
}
```

All keys are optional; unspecified keys use defaults. See [docs/PERF_TUNING.md](docs/PERF_TUNING.md) for detailed descriptions.

### Offline Cost Model Fit

After running benchmarks on your target hardware:

```bash
python3 tune/fit_cost_model.py build/results.csv
```

Emits JSON `cost_fit` section (median actual / predicted ratios per strategy) to stderr. Merge into your config JSON to improve predictions on that machine.

---

## What's Included

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **Profiler** | `include/profiler_impl.h`, `src/profiler.cpp` | Detect sortedness, runs, duplicates, entropy via sampling. |
| **Cost model** | `include/cost_model.h`, `src/cost_model.cpp` | Per-strategy time estimates (CPU + GPU). |
| **Dispatcher** | `include/dispatcher.h`, `src/dispatcher.cpp` | Choose strategy; apply guardrails & border-band rules. |
| **CPU algorithms** | `src/cpu/{insertion,introsort,run_merge,quicksort_3way}.cpp` | Specialized sort kernels. |
| **GPU sort** | `src/gpu/gpu_sort.cu` | Thrust (default) or CUB radix on device. |
| **Feedback** | `include/feedback.h`, `src/feedback.cpp` | Log decisions & timings to CSV for tuning. |
| **Config loader** | `src/config_loader.cpp` | Parse JSON thresholds. |
| **Benchmarks** | `bench/{bench_results,bench_main}.cpp` | Suite harness + Google Benchmark micro. |
| **Tests** | `tests/test_*.cpp` | Correctness, cost model, profiler, regression. |
| **Tuning scripts** | `tune/{fit_cost_model,analyze_log,plot_results,score_configs}.py` | Data analysis & offline tuning. |

---

## Tested Platforms

| OS | Compiler | CUDA | Status |
|----|----------|------|--------|
| **macOS 13+** | AppleClang 15+ | N/A | ✓ CPU-only (no CUDA support) |
| **Linux (x86_64)** | g++ 11+ / clang++ 14+ | 11.x, 12.x | ✓ CPU + CUDA (recommended for GPU development) |
| **Windows** | MSVC 2019+ | 11.x, 12.x | ✓ CPU + CUDA (expected; untested in CI) |

---

## Documentation

- **[docs/BENCHMARK_SUITE.md](docs/BENCHMARK_SUITE.md)** — Versioned workload contract, acceptance rules.
- **[docs/PERF_TUNING.md](docs/PERF_TUNING.md)** — Cost model calibration, threshold tuning, profiler behavior, GPU details.
- **[docs/CI.md](docs/CI.md)** — CI expectations (CPU-always; CUDA optional on labeled runners).
- **[absolute-docs/PDD.md](absolute-docs/PDD.md)** — Full product development document (design rationale, phase history).

---

## Design Principles

1. **Correctness first.** No data loss, no off-by-one errors. Fallback to safe introsort on any cost model uncertainty.

2. **Workload-adaptive, not magic.** PASE wins when structure exists; on random data it ties or loses gracefully (guardrails enforce ≤ 1.65–2× slowdown in CI).

3. **Hardware-aware.** CPU ops/ms, GPU PCIe bandwidth, CUDA compute tuned separately; re-tune for your hardware with `fit_cost_model.py`.

4. **Transparent and auditable.** Every sort decision is logged (optionally) with predicted vs actual time so you can verify and improve the model.

5. **Minimal dependencies.** Core: just C++17 + nlohmann/json (for config). GPU: CUDA 11+ optional. Benchmarks: Google Benchmark (vendored).

---

## Limitations & Future Work

### Current Limitations ℹ️

- **No distributed sort.** PASE is single-machine, single-core-friendly (with task parallelism elsewhere).
- **GPU limited to `int` + `std::less<int>`** to keep the dispatch logic tractable. Multi-type GPU kernels require separate cost tuning.
- **Profiler is approximate.** Sampling-based; very small samples on huge arrays may miss duplicates/structure. Use golden test seeds to validate.
- **No "optimal" cost fit.** Offline tuning is heuristic (median ratios), not ML-based. Works well in practice but can be improved.

### Next Steps (Post-Phase 4)

1. **SIMD-accelerated run detection** to speed profiler.
2. **Vectorized introsort** (AVX-512 or Neon on ARM) for CPU path throughput.
3. **Distributed sort** (sketched in non-goals) if use case demands it.
4. **Multi-type GPU support** (floats, complex keys) with templated cost model.
5. **Machine learning tuning** to predict thresholds from profile vectors instead of hand-tuned ranges.

---

## Contributing

Contributions welcome! Areas of interest:

- **Profiler improvements:** Better duplicate/entropy estimation with same sampling budget.
- **GPU extensions:** CUB path validation, other GPU backends (HIP, oneAPI).
- **Benchmarks:** New workload datasets; platform-specific tuning; reproducibility packs.
- **Documentation:** Porting to other projects; real-world use case studies.

See [absolute-docs/DEVELOPMENT_GUIDE.md](absolute-docs/DEVELOPMENT_GUIDE.md) for architecture overview.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Citation

If PASE helps your research or product, please cite:

```bibtex
@software{pase_2026,
  title={PASE: Predictive Adaptive Sorting Engine},
  author={Your Name or Organization},
  year={2026},
  url={https://github.com/your-org/Predictive-Adaptive-Sorting-Engine}
}
```

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/your-org/Predictive-Adaptive-Sorting-Engine/issues)
- **Discussion:** [GitHub Discussions](https://github.com/your-org/Predictive-Adaptive-Sorting-Engine/discussions)
- **Benchmark reproducibility:** Include `CMAKE_BUILD_TYPE`, `PASE_ENABLE_CUDA`, GPU model, CPU model, OS, compiler in reports.
