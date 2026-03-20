# PASE benchmark suite contract (v1.1)

Identifier: **`PASE_BENCH_SUITE_VERSION`** = `"1.1"` (see `include/pase_bench_contract.h`).

## Workloads

Datasets from `bench/gen_datasets.h`:

| id | description |
|----|-------------|
| sorted | ascending |
| reverse | descending |
| nearly_sorted_95 / 80 | iota + random swaps |
| random | shuffled iota |
| heavy_dup | uniform 0..9 |
| clustered | cluster * 1000 + offset |
| long_runs | concatenated ascending runs (timsort / run-merge stress) |
| pipe_organ | ascending then descending halves |
| large_random_1M / 10M | forced sizes (full bench only) |

## Default size grids

- **Quick** (`--quick`): single size **100,000**; skips 1M/10M types.
- **Full** (default): **10,000**, **100,000**, **500,000** per dataset × size cell.

Override with `--sizes N1,N2,...`.

## Baselines

Each cell records mean ± stdev ms for:

1. PASE `adaptive_sort`
2. `std::sort`
3. `std::stable_sort`

Derived columns: `speedup_vs_std`, `speedup_vs_stable`.

## CSV

- Optional first line: `# pase_bench_suite=1.1` (version marker).
- Schema is stable within a major suite version; bump version if columns change.

## Acceptance (reference; tune in CI)

See `pase::bench_contract` in `include/pase_bench_contract.h`:

- **Structured** workloads (sorted / nearly sorted): PASE median wall time should not exceed **`kAcceptStructuredMaxSlowdown` ×** `std::sort` median in `PerformanceRegression` tests.
- **Random / general**: looser **`kAcceptRandomMaxSlowdown`** for optional extended smoke (not all scenarios need to “win”).

Document machine model, OS, `CMAKE_BUILD_TYPE`, and `PASE_ENABLE_CUDA` when publishing numbers.
