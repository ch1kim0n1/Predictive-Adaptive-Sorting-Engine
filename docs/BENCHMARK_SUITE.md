# PASE benchmark suite contract (v1.2)

Identifier: **`PASE_BENCH_SUITE_VERSION`** = `"1.2"` (see `include/pase_bench_contract.h`).

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

- Optional first line: `# pase_bench_suite=1.2` (version marker).
- Schema is stable within a major suite version; bump version if columns change.

## Acceptance (reference; tune in CI)

See `pase::bench_contract` in `include/pase_bench_contract.h` and `tests/test_performance_regression.cpp`:

| Constant | Use |
|----------|-----|
| **`kAcceptFullySortedMaxSlowdown`** | Fully sorted **100k** `int`: PASE median vs `std::sort` (profiler + dispatch can dominate very fast libc sorts on some hosts). |
| **`kAcceptStructuredMaxSlowdown`** | **Nearly sorted (95%)** @ 100k: PASE median vs `std::sort`. |
| **`kAcceptRandomMaxSlowdown`** | **Random** smoke (50k): looser bound; not every scenario must “win”. |

Document machine model, OS, `CMAKE_BUILD_TYPE`, and `PASE_ENABLE_CUDA` when publishing numbers.
