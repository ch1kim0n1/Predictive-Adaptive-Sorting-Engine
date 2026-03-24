# PASE Visuals — Benchmark Charts & Methodology

This folder contains benchmark visualization assets for the
**Predictive Adaptive Sorting Engine (PASE)**.

---

## Contents

| File | Description |
|------|-------------|
| [`speedup_chart.png`](speedup_chart.png) | Bar chart — PASE speedup vs `std::sort` across all workload types and array sizes |
| [`cost_model_fit.png`](cost_model_fit.png) | Line chart — PASE vs `std::sort` sort latency (ms) across array sizes; illustrates where the cost model dispatches to a specialist or GPU strategy |
| [`mock_bench_results.csv`](mock_bench_results.csv) | Representative benchmark data (based on README v1.2 performance claims) used to generate the charts above |
| [`generate_visuals.py`](generate_visuals.py) | Reproducible Python script — regenerates both PNGs from any compatible CSV |

---

## Performance Summary (PASE v1.2)

The charts above visualise the following performance claims from the README:

| Workload | Speedup vs `std::sort` | Strategy Dispatched |
|----------|------------------------|---------------------|
| **Fully sorted** | **1.5–2.0×** | Insertion-optimised path (O(n) early bailout) |
| **Nearly sorted (95%)** | **1.2–1.6×** | Run-merge with galloping merge |
| **Heavy duplicates** | **1.3–2.0×** | 3-way quicksort `[<x][==x][>x]` partition |
| **Long structured runs** | **1.4–1.8×** | Run-merge / timsort-like |
| **Large data (n ≥ 100 k, GPU)** | **2–8×** | CUDA Thrust/CUB device sort |
| **Fully random** | ~1.0× (ties) | Introsort fallback (overhead < 3%) |

---

## Benchmarking Methodology

### 1 — Build the benchmark binary

```bash
# From the repository root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_results -j$(nproc)
```

GPU support (optional):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPASE_ENABLE_CUDA=ON
cmake --build build --target bench_results -j$(nproc)
```

### 2 — Run the full benchmark suite

```bash
# Quick run (10 k & 100 k only)
./build/bench/bench_results --quick --out visuals/bench_results.csv

# Full suite (10 k / 100 k / 500 k × 8 dataset types)
./build/bench/bench_results --out visuals/bench_results.csv
```

The CSV will contain columns:
`dataset, n, pase_ms, std_ms, std_stable_ms, speedup_vs_std, strategy`

Comment lines beginning with `#` (emitted by `pase_bench_suite`) are
automatically stripped by both `generate_visuals.py` and `tune/pase_csv.py`
before parsing.

### 3 — Generate charts

```bash
# Using this script (reads mock_bench_results.csv by default)
python3 visuals/generate_visuals.py

# Using live benchmark results
python3 visuals/generate_visuals.py --csv visuals/bench_results.csv --out-dir visuals/

# Alternatively use the tune/ script directly
python3 tune/plot_results.py visuals/bench_results.csv -o visuals/bench_speedup.png
```

Install Python dependencies if needed:
```bash
pip install -r tune/requirements.txt
```

### 4 — Chart descriptions

#### `speedup_chart.png`
- **X-axis:** workload type and array size (e.g., `sorted (n=100k)`)
- **Y-axis:** speedup factor relative to `std::sort` (1.0 = baseline)
- **Bar colour:** strategy dispatched by the PASE cost model
  - 🔵 Blue — Insertion sort (sorted / nearly-sorted short arrays)
  - 🟢 Green — Run-merge (structured runs)
  - 🟠 Orange — 3-way quicksort (heavy duplicates)
  - ⚫ Grey — Introsort fallback (random / mixed data)
  - 🔴 Pink/red — GPU (CUDA Thrust/CUB, large n)
- **Dashed red line:** `std::sort` baseline at 1.0×

#### `cost_model_fit.png`
- **X-axis:** array size `n` (log scale)
- **Y-axis:** sort latency in milliseconds (log scale)
- **Solid lines:** PASE latency per dataset
- **Dashed lines:** `std::sort` latency per dataset (same colour, faded)
- **Pink dotted vertical line:** GPU dispatch threshold (n ≥ 100 k)

A well-calibrated cost model produces solid lines that stay **below** their
dashed counterparts across the relevant size range.  For random data the two
lines overlap (PASE falls back to introsort).

---

## Tuning & Calibration

The cost model coefficients live in `tune/optimized_thresholds.json` (generated
by `tune/fit_cost_model.py`).  After running benchmarks on new hardware:

```bash
python3 tune/fit_cost_model.py visuals/bench_results.csv \
    --out tune/optimized_thresholds.json
```

See [`docs/PERF_TUNING.md`](../docs/PERF_TUNING.md) for full details.

---

## Notes

- The current PNGs were generated from **mock data** (`mock_bench_results.csv`)
  that faithfully represents the v1.2 performance claims in the README.
  They are intended as reference charts and for documentation/LinkedIn showcase.
- To produce **live** charts from real hardware measurements, follow steps 1–3
  above and commit the resulting PNGs in place of these placeholders.
- The `generate_visuals.py` script is fully compatible with the CSV format
  produced by `bench/bench_results` and `tune/plot_results.py`.
