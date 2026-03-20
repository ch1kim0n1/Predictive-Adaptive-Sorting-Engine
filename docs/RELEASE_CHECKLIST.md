# Release checklist (PASE)

Use this before tagging a release (e.g. `v1.x.y`).

## 1. Working tree

- [ ] `git status` clean or only intentional changes; no broken doc links in `README.md`.
- [ ] `docs/BENCHMARK_SUITE.md` and `docs/PERF_TUNING.md` present and aligned with `PASE_BENCH_SUITE_VERSION` in `include/pase_bench_contract.h`.
- [ ] `docs/PHASE5.md` and README Phase 5 table agree (features + limitations).

## 2. Documentation reconcile

- [ ] README “acceptance bounds” / design principles match the three `kAccept*MaxSlowdown` constants.
- [ ] `docs/CI.md` lists the CPU tests you run by default; optional MPI/CUDA jobs documented.

## 3. Build & test matrix

Run from a fresh build directory or `cmake --build` after configure.

### CPU (required)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DPASE_ENABLE_MPI=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expect **8** default tests (includes `Phase5Integration`, `GpuComplexSort`, `PerformanceRegression`).

### MPI (when shipping MPI support)

Requires MPI installed (e.g. Homebrew `open-mpi`, Linux `libopenmpi-dev`).

```bash
rm -rf build-mpi
cmake -B build-mpi -S . -DCMAKE_BUILD_TYPE=Release -DPASE_ENABLE_MPI=ON
cmake --build build-mpi -j
ctest --test-dir build-mpi --output-on-failure
```

Confirm **`MpiPhase5GlobalSort`** runs (`mpirun -n 2`).

### CUDA (when shipping GPU paths)

On a machine with NVIDIA driver + toolkit:

```bash
rm -rf build-cuda
cmake -B build-cuda -S . -DCMAKE_BUILD_TYPE=Release \
  -DPASE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

GPU tests skip if no device; they should **pass** (or skip) without failure.

## 4. Tag & publish

```bash
git tag -a vX.Y.Z -m "PASE vX.Y.Z"
git push origin vX.Y.Z
```

- [ ] Update **GitHub release notes** (or keep `docs/RELEASE_NOTES_PHASE5.md` as engineering summary and paste summary into the GitHub release body).

## Notes

- If **`PerformanceRegression.Sorted100kNotPathological`** fails on a new host, see `kAcceptFullySortedMaxSlowdown` in `pase_bench_contract.h` — fully sorted large arrays stress profiler overhead vs very fast `std::sort`.
