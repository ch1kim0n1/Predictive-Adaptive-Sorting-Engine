# PASE performance tuning

## Evaluation contract (versioned suite)

- Suite id **`PASE_BENCH_SUITE_VERSION`** and workload list: see
  [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md) and `include/pase_bench_contract.h`.
- Regression tests (`tests/test_performance_regression.cpp`) compare PASE median
  wall time vs `std::sort` using:
  - **`kAcceptFullySortedMaxSlowdown`** — fully sorted **100k** `int`
  - **`kAcceptStructuredMaxSlowdown`** — **nearly sorted (95%)** @ 100k
  - **`kAcceptRandomMaxSlowdown`** — random **50k** smoke
- CSV exports from `bench_results` start with `# pase_bench_suite=…`; Python
  tooling strips comment lines automatically.

## 1. Benchmark harness

From `build/`:

```bash
./bench/bench_results --out results.csv --sizes 10000,100000,500000 --repeat 9
python3 ../tune/analyze_log.py results.csv --col speedup_vs_std
python3 ../tune/score_configs.py results.csv
```

## 2. Dispatcher thresholds (`PASE_CONFIG` JSON)

Keys: `sorted`, `run_merge`, `dup`, `min_gpu`, `gpu_win_factor`, `max_insertion_n`,
`strategy_guardrail`, `gpu_rel_margin`, and **border-band conservatism** (CPU
dispatch near dup / run-merge thresholds):

- `dup_border_band` (float, default `0.08`)
- `run_merge_border` (int, default `6`)
- `conservative_specialist_frac` (float, default `0.88`) — specialist CPU
  strategy must beat introsort by this margin when the profile sits in the
  border band; otherwise **INTROSORT** is chosen.

See README and `include/dispatcher.h`.

## 3. Offline cost-model fit (`cost_fit`)

After collecting `bench_results.csv` with columns `strategy`, `pase_ms`, `pred_cpu_ms`:

```bash
python3 tune/fit_cost_model.py results.csv
```

Paste the printed `"cost_fit"` object into your JSON config, e.g.:

```json
{
  "sorted": 0.9,
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

**Semantics**: scales multiply internal model terms so predicted ms tracks observed
strategy times (heuristic medians, **clamped per key to \[0.2, 5.0\]** in
`fit_cost_model.py`). The script prints **clamp / bounds guidance** on stderr when
scales hit limits. `gpu_kernel` scales the GPU **compute**
term only (PCIe transfer model unchanged).

## 4. GPU path (CUDA builds)

- **Default**: **Thrust** `thrust::sort` on device for `int`, `float`, `double`, and
  lexicographic **complex** (`gpu_sort_complex_*`: pack to `(re, im)` POD, sort, unpack).
- **Optional** (`-DPASE_GPU_SORT_USE_CUB=ON`): **CUB** `DeviceRadixSort` on
  XOR-mapped `int` keys (same `gpu_sort_int` API).

Minimum practical `n` is ~8k elements; the dispatcher also gates on `min_gpu`
and `gpu_rel_margin`.

Feedback CSV (`~/.pase/sort_log.csv`) logs **`pred_gpu_transfer_ms`** and
**`pred_gpu_kernel_ms`** separately (sum matches the combined GPU estimate used
for dispatch).

## 5. Profiler duplicate signal

`duplicate_ratio` blends **adjacent sample pairs** with **sorted-sample adjacency**
so heavy duplicate distributions register even when duplicates are not neighbors
in the original array.

**Second-stage sampling** (mid-stride indices) runs only when **stride-only**
metrics look **ambiguous** (duplicate signal in a middle band, or sortedness
neither very high nor very low), keeping cost low when the first pass already
indicates clear structure.
