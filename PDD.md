PASE — Predictive Adaptive Sorting Engine     |     Product Development Document v1.0


PRODUCT DEVELOPMENT DOCUMENT
Predictive Adaptive Sorting Engine
CPU + CUDA Hybrid — Self-Tuning Runtime Architecture






Project Code
	PASE-v1.0
	Status
	Active Development
	Classification
	Portfolio / Open Source
	Document Version
	1.0 — Initial Release
	Date
	March 2026
	Owner
	Vlad — MindCore LLC






Plan:


* Build full system


* Benchmark aggressively


* Open-source it


* Write strong README + blog


Outcome:


* Extremely strong resume signal


* Interview leverage


* Hackathon / demo ready


Verdict:


* Guaranteed payoff
	



________________


1. Executive Summary




PASE (Predictive Adaptive Sorting Engine) is a systems-level sorting framework that combines runtime data profiling, cost-model-driven dispatch, adaptive CPU algorithm selection, and CUDA-accelerated GPU kernels into a single unified pipeline. Unlike a sorting algorithm, PASE is a sorting system — one that observes, reasons, and self-tunes.


The project is designed to be portfolio-defining: it demonstrates algorithm design, memory hierarchy engineering, GPU programming, performance benchmarking, and an upgrade path toward ML-driven dispatch — all in a single coherent codebase.


What separates PASE from every other sorting project
	1. Sublinear profiler — samples 1-2% of the array for O(n) behavior with near-zero overhead
	2. Cost model dispatcher — estimates GPU transfer cost vs predicted CPU time before committing
	3. Feedback loop — logs {profile, strategy, actual runtime} and tunes thresholds automatically
	4. Human-readable diagnostics — --verbose flag explains every dispatch decision in plain English
	5. Workload-characterization benchmarks — sortedness score vs speedup curves, not just raw timings
	

	

Resume-Ready Framing
Project title for resume / LinkedIn / GitHub:


Predictive Adaptive Sorting Engine (CPU + CUDA Hybrid)
	- Designed self-tuning sorting system using runtime data profiling to dynamically select optimal algorithms
	- Built CUDA kernels (bitonic + parallel merge) with shared-memory optimization and coalesced access
	- Implemented cost-model-driven GPU dispatch, estimating PCIe transfer + kernel time vs CPU prediction
	- Achieved X% speedup over std::sort / Timsort on structured and high-entropy workloads
	- Built feedback loop logging {profile vector, strategy, runtime} to auto-tune dispatch thresholds
	- Benchmarking suite: cache miss analysis (perf stat), GPU occupancy, sortedness-vs-speedup curves
	

	

________________


2. System Architecture




PASE is structured as five tightly coupled subsystems. Each subsystem has a single responsibility and a well-defined interface to the next.


2.1  High-Level Pipeline


Input Array
    |
    v
[Sampling Profiler]        <- O(n * sample_rate), cache-friendly, branch-light
    |
    v  Profile struct
[Cost Model]               <- estimates GPU vs CPU predicted runtime
    |
    v  Strategy enum
[Execution Dispatcher]
    |
    +---> [CPU Path]       <- insertion / run-extension / cache-aware merge / introsort / 3-way QS
    |
    +---> [GPU Path]       <- bitonic block sort + parallel merge (CUDA)
    |
    v
[Feedback Logger]          <- writes {profile, strategy, actual_ns} to runtime log
    |
    v
Sorted Output


2.2  Subsystem Responsibility Map


Subsystem
	Responsibility
	Key Output
	Sampling Profiler
	Single-pass scan over 1-2% sample; computes sortedness, entropy, dup ratio, run lengths
	Profile struct
	Cost Model
	Estimates expected runtime for each candidate strategy including PCIe overhead for GPU
	Strategy enum + confidence
	Execution Dispatcher
	Routes to correct CPU or GPU path based on cost model output
	Calls execution path
	CPU Execution Engine
	Optimized insertion, run-extension merge, cache-tiled merge, introsort, 3-way quicksort
	Sorted array (in-place)
	GPU Execution Engine
	Bitonic block sort in shared memory, parallel merge passes, async streams
	Sorted array (host)
	Feedback Logger
	Logs profile vector + chosen strategy + measured runtime after each sort
	Runtime log (CSV/SQLite)
	Threshold Tuner
	Offline grid-search or online EMA over feedback log to re-tune dispatch thresholds
	Updated config file
	

________________


3. Sampling Profiler




The profiler is the most original component of PASE. Instead of scanning the full array, it samples a configurable fraction (default 1.5%) and computes six metrics that fully characterize the input workload. Profiling cost is sublinear, which matters for very large arrays.


3.1  Profile Struct (C++)
struct Profile {
    float  sortedness;        // 0.0 = fully random, 1.0 = perfectly sorted
    float  duplicate_ratio;   // fraction of elements that are duplicates
    float  entropy;           // Shannon entropy normalized to [0, 1]
    int    avg_run_length;    // average ascending run length in sample
    int    max_run_length;    // longest ascending run in sample
    float  value_spread;      // (max - min) / n, normalized range
    int    n;                 // full array size
    float  sample_rate;       // fraction actually scanned
};


3.2  Metric Definitions


Metric
	How computed
	Why it matters
	sortedness
	Count adjacent pairs in order / total adjacent pairs in sample
	Primary dispatch signal
	duplicate_ratio
	Count elements matching predecessor in sorted sample bucket
	Routes to 3-way QS
	entropy
	Shannon entropy over histogram of sampled values
	High entropy => GPU wins
	avg_run_length
	Mean run length across ascending streaks in sample
	Long runs => run-merge wins
	max_run_length
	Longest single ascending streak in sample
	Near-sorted detection
	value_spread
	(max - min) / n normalized
	Signals integer vs float domain
	

3.3  Implementation Notes
* Single cache-friendly forward scan — no random access, no branch-heavy inner loops
* Sample indices are evenly spaced (stride = n / sample_count) to avoid clustering bias
* Runs computed incrementally in one pass; no auxiliary array needed
* Entropy estimated via 256-bucket histogram built during same pass (O(sample_size))
* Total profiling cost: O(n * sample_rate) time, O(256) extra space


3.4  Diagnostic Output (--verbose)
[PASE Profiler]
  n               = 2,000,000
  sample_rate     = 1.5%  (30,000 elements scanned)
  sortedness      = 0.94
  avg_run_length  = 312
  duplicate_ratio = 0.03
  entropy         = 0.21  (low — structured data)


[PASE Cost Model]
  GPU est.        = 18.4 ms  (incl. 11.2 ms PCIe transfer)
  CPU est.        = 6.7 ms   (run-merge path)
  Decision        => RUN_MERGE_OPT  (CPU 2.7x cheaper)


This diagnostic output is the demo moment in interviews. It makes the system explainable and shows the cost model reasoning transparently.


________________


4. Cost Model & Dispatcher




The cost model is the engineering insight that elevates PASE above every student sorting project. Instead of a hard-coded threshold ("use GPU if n > 500k"), PASE estimates the actual expected runtime for each strategy before committing.


4.1  GPU Cost Estimation
double estimate_gpu_cost(int n, float entropy) {
    // PCIe transfer: ~12 GB/s bidirectional
    double bytes          = n * sizeof(int);
    double transfer_ms    = (2.0 * bytes / 12e9) * 1000.0;  // H2D + D2H


    // Bitonic sort: O(n log^2 n) ops at ~5 TFLOPS effective throughput
    double ops            = n * log2(n) * log2(n);
    double kernel_ms      = ops / 5e12 * 1000.0;


    // Entropy factor: high entropy -> less branch divergence -> faster kernel
    double entropy_factor = 1.0 - 0.3 * entropy;


    return transfer_ms + kernel_ms * entropy_factor;
}


4.2  CPU Cost Estimation
double estimate_cpu_cost(Profile& p, Strategy s) {
    double base;
    if (s == RUN_MERGE_OPT) {
        // Merge cost scales with inversion count, not n log n
        base = p.n * log2((double)p.n / p.avg_run_length);
    } else if (s == INSERTION_OPT) {
        base = p.n * (1.0 - p.sortedness) * p.n * 0.5;  // quadratic reduced by sortedness
    } else {
        base = p.n * log2(p.n);  // introsort / 3-way QS fallback
    }
    return base / CPU_THROUGHPUT_OPS_PER_MS;  // calibrated at startup
}


4.3  Dispatcher Logic
Strategy select_strategy(Profile& p) {
    // Fast paths — no cost model needed
    if (p.sortedness > THR.sorted)         return INSERTION_OPT;
    if (p.sortedness < (1.0-p.sortedness)
        && p.n < THR.min_gpu)              goto cpu_path;


    // Cost model arbitration
    double gpu_cost = estimate_gpu_cost(p.n, p.entropy);
    double cpu_cost = estimate_cpu_cost(p, best_cpu_strategy(p));


    if (gpu_cost < cpu_cost * 0.85)        return GPU_SORT;  // 15% margin


cpu_path:
    if (p.avg_run_length > THR.run_merge)  return RUN_MERGE_OPT;
    if (p.duplicate_ratio > THR.dup)       return THREE_WAY_QS;
    return INTROSORT;
}


Design note — the 0.85 margin
	GPU dispatch is penalized by a 15% margin to account for CUDA driver latency variability.
	This prevents thrashing at the threshold boundary where GPU and CPU costs are nearly equal.
	The margin value is a tunable threshold that the feedback loop can adjust automatically.
	

	

________________


5. CPU Execution Engine




The CPU engine must outperform Timsort on specific workload niches. The strategy: beat Timsort exactly where it is weakest — nearly-sorted arrays with micro-disorder, and high-duplicate arrays where comparison cost dominates.


5.1  Algorithm Selection Matrix


Strategy
	When used
	Key optimization
	vs Timsort
	INSERTION_OPT
	sortedness > 0.90
	Binary search + memmove, unrolled
	Wins: micro-disorder
	RUN_MERGE_OPT
	avg_run_length > THR.run_merge
	Run extension + cache-tiled merge
	Wins: structured data
	THREE_WAY_QS
	duplicate_ratio > 0.40
	Fat-partition avoids re-comparison
	Wins: heavy duplicates
	INTROSORT
	Fallback for random data
	Heapsort fallback prevents O(n^2)
	Parity on random
	GPU_SORT
	Cost model: GPU cheaper
	Bitonic + parallel merge on CUDA
	Wins: large random
	

5.2  Optimized Insertion Sort
Binary search for insertion position reduces comparisons from O(n) to O(log n) per element. memmove replaces individual swaps, leveraging SIMD-optimized libc. Loop unrolling reduces branch overhead on modern OOO CPUs.


void insertion_optimized(int* A, int n) {
    for (int i = 1; i < n; i++) {
        int key = A[i];
        // Binary search for insertion index
        int lo = 0, hi = i;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (A[mid] <= key) lo = mid + 1;
            else               hi = mid;
        }
        // Shift right with memmove (SIMD-accelerated in libc)
        memmove(&A[lo + 1], &A[lo], (i - lo) * sizeof(int));
        A[lo] = key;
    }
}


5.3  Run Detection + Extension (the PASE edge)
Timsort detects natural runs and merges them. PASE goes further: it detects micro-disordered regions within runs and locally repairs them before they become expensive merge inputs. A run with 3 out-of-place elements is cheaper to fix inline than to break into separate merge inputs.


// Extend a run rightward, fixing small local disorder inline
int extend_run(int* A, int start, int n, int repair_budget) {
    int end = start + 1;
    while (end < n) {
        if (A[end] >= A[end-1]) {
            end++;  // normal ascending continuation
        } else if (repair_budget > 0 && end + 1 < n && A[end+1] >= A[end-1]) {
            // One-element disorder: swap and continue
            swap(A[end], A[end-1]);
            repair_budget--;
            end++;
        } else {
            break;  // genuine run boundary
        }
    }
    return end;  // exclusive end of extended run
}


5.4  Cache-Aware Merge
Standard merge sort makes full-array passes, causing L3 cache thrash on large arrays. PASE tiles its merge to fit the working set into L2 cache. The tile size is calibrated at startup by measuring actual L2 size from /sys or cpuid.


// Merge two adjacent sorted runs, tiled to L2 cache
// tile_size = L2_bytes / sizeof(int) / 2  (half for each side)
void cache_tiled_merge(int* A, int lo, int mid, int hi, int tile_size) {
    for (int t = lo; t < hi; t += tile_size) {
        int t_end = min(t + tile_size, hi);
        // Standard merge within this tile only
        // Both input tiles fit in L2; no cache miss overhead
        merge_range(A, t, mid, t_end);
    }
}


Key metric to report in benchmarks: cache miss rate with perf stat --event cache-misses,cache-references. Expect 30-50% reduction vs naive mergesort on arrays > 4MB.


________________


6. GPU Execution Engine (CUDA)




The GPU engine is where PASE becomes genuinely competitive for large, high-entropy arrays. The design uses a two-phase pipeline: block-level bitonic sort in shared memory, followed by parallel merge passes across blocks. CUDA streams are used to overlap PCIe transfer with compute.


6.1  GPU Pipeline
Host Array
    |
    | cudaMemcpyAsync (stream 0)  -- overlap with CPU preprocessing
    v
Device Array (global memory)
    |
    v
[Phase 1: Block-Level Bitonic Sort]
    - blockDim = 1024 threads per block
    - Each block sorts its 1024 elements entirely in shared memory
    - No global memory traffic during sort
    - Result: n/1024 sorted blocks of size 1024
    |
    v
[Phase 2: Parallel Merge Passes]
    - Pass k: merge pairs of blocks of size 2^k into blocks of size 2^(k+1)
    - Each pass: one kernel launch, full coalesced global reads/writes
    - log2(n/1024) passes total
    |
    | cudaMemcpyAsync (stream 0)  -- device to host
    v
Sorted Host Array


6.2  Block Sort Kernel
__global__ void blockBitonicSort(int* data, int n) {
    __shared__ int shm[BLOCK_SIZE];  // BLOCK_SIZE = 1024


    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;


    // Load into shared memory (coalesced global read)
    shm[tid] = (idx < n) ? data[idx] : INT_MAX;
    __syncthreads();


    // Bitonic sort network — all in shared memory
    for (int k = 2; k <= BLOCK_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                if ((ascending && shm[tid] > shm[ixj]) ||
                   (!ascending && shm[tid] < shm[ixj])) {
                    int tmp = shm[tid];
                    shm[tid] = shm[ixj];
                    shm[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }


    // Write back (coalesced global write)
    if (idx < n) data[idx] = shm[tid];
}


6.3  CUDA Optimization Checklist


Optimization
	Implementation
	Impact
	Coalesced memory access
	Consecutive threads access consecutive addresses in global mem
	Max memory bandwidth
	Shared memory usage
	All bitonic comparisons in __shared__; zero global traffic
	Eliminates L2 pressure
	Warp divergence avoidance
	Bitonic network: all threads take same branch per step
	Full warp utilization
	CUDA streams
	cudaMemcpyAsync overlaps H2D transfer with CPU profiling
	Hides PCIe latency
	INT_MAX padding
	Pad to power-of-2 with INT_MAX; sorts correctly, merges cleanly
	Avoids bounds checks
	cudaOccupancyMaxPotentialBlockSize
	Auto-tune block size per GPU at runtime
	Maximizes occupancy
	

6.4  CPU-GPU Concurrent Execution (Advanced)
For very large arrays, PASE can sort the first half on GPU while the CPU sorts the second half, then merge. This is the most advanced feature and should be implemented in Phase 3B.


// Split-and-conquer: GPU left half, CPU right half, then merge
void concurrent_sort(int* A, int n) {
    int mid = n / 2;


    // Launch GPU sort on A[0..mid] asynchronously
    cudaMemcpyAsync(d_A, A, mid * sizeof(int), H2D, stream0);
    blockBitonicSort<<<blocks, 1024, 0, stream0>>>(d_A, mid);


    // CPU sorts A[mid..n] in parallel (this thread)
    introsort(A + mid, n - mid);


    // Wait for GPU, copy back, then merge
    cudaStreamSynchronize(stream0);
    cudaMemcpyAsync(A, d_A, mid * sizeof(int), D2H, stream0);
    cudaStreamSynchronize(stream0);


    // Final merge of two sorted halves
    cache_tiled_merge(A, 0, mid, n, L2_TILE_SIZE);
}


________________


7. Feedback Loop & Self-Tuning




This section is the biggest architectural upgrade over the original plan. Instead of hand-tuned static thresholds, PASE logs every sort decision and automatically adjusts thresholds from empirical data. This is what makes it genuinely adaptive rather than just rule-based.


7.1  What Gets Logged
// Log entry written after every sort completes
struct SortLog {
    // Profile features (inputs to dispatcher)
    float  sortedness;
    float  duplicate_ratio;
    float  entropy;
    int    avg_run_length;
    int    n;


    // Decision
    Strategy chosen_strategy;
    double  predicted_cpu_ms;
    double  predicted_gpu_ms;


    // Actual outcome
    double  actual_ms;
    bool    prediction_correct;   // was chosen strategy actually fastest?
};


// Written to: ~/.pase/sort_log.csv   (or SQLite for Phase 4)


7.2  Online Threshold Tuning (Phase 3)
Exponential moving average applied to per-strategy runtime observations. After every sort, the threshold that governs dispatching to that strategy is nudged toward the empirical breakeven point.


// After each sort, update the relevant threshold with EMA
void update_threshold(float& threshold, float actual_cost,
                       float predicted_cost, float alpha=0.05) {
    float error = actual_cost - predicted_cost;
    // If we underpredicted, make threshold more conservative (higher margin)
    threshold += alpha * error;
    threshold = clamp(threshold, THR_MIN, THR_MAX);
}


7.3  Offline Threshold Optimization (Phase 4)
After collecting 1,000+ log entries, an offline optimizer runs grid search across threshold space to minimize total misprediction cost. This is the upgrade path toward an ML-based selector.
# Python: offline grid search over threshold space
# Input: sort_log.csv
# Output: optimized_thresholds.json


import pandas as pd, numpy as np, json


log = pd.read_csv('~/.pase/sort_log.csv')


best_cost = float('inf')
best_thresholds = {}


for thr_sorted in np.linspace(0.80, 0.98, 20):
  for thr_run in range(32, 256, 16):
    for thr_dup in np.linspace(0.20, 0.60, 10):
      # Simulate dispatch on log with these thresholds
      predicted = simulate_dispatch(log, thr_sorted, thr_run, thr_dup)
      cost = total_regret(log, predicted)  # actual_ms - optimal_ms summed
      if cost < best_cost:
        best_cost = cost
        best_thresholds = {'sorted': thr_sorted,
                           'run_merge': thr_run, 'dup': thr_dup}


json.dump(best_thresholds, open('optimized_thresholds.json','w'))
print(f'Best total regret: {best_cost:.2f} ms')


________________


8. Benchmarking Plan




Benchmarking is not optional and not an afterthought. The benchmark suite is part of the product. The goal is not just to show PASE is fast — it is to characterize exactly when and why it wins, and to produce figures that can be included in a README or talked through in an interview.


8.1  Input Datasets


Dataset
	Description
	Expected PASE winner
	sorted
	Fully ascending array
	INSERTION_OPT
	reverse
	Fully descending array
	INSERTION_OPT (with reverse detection)
	nearly_sorted_95
	95% sorted, 5% random swaps
	RUN_MERGE_OPT
	nearly_sorted_80
	80% sorted — harder case
	RUN_MERGE_OPT or INTROSORT
	random
	Uniformly random integers
	GPU (large n), INTROSORT (small n)
	heavy_dup
	~50% duplicate values
	THREE_WAY_QS
	clustered
	Values drawn from 5 tight clusters
	THREE_WAY_QS or RUN_MERGE
	pipe_organ
	Ascending then descending (Timsort nightmare)
	RUN_MERGE_OPT
	large_random_1M
	1M random ints — GPU benchmark
	GPU_SORT
	large_random_10M
	10M random ints — max GPU benchmark
	GPU_SORT
	

8.2  Comparison Targets
* std::sort (C++) — introsort baseline
* std::stable_sort (C++) — merge-sort baseline
* Python sorted() — Timsort reference
* thrust::sort (CUDA Thrust) — GPU baseline
* Intel oneTBB parallel sort — optional stretch comparison


8.3  Metrics to Report


Metric
	Tool
	What it proves
	Wall-clock runtime (ms)
	std::chrono, CUDA events
	Primary performance claim
	Cache miss rate
	perf stat -e cache-misses
	Cache-aware merge actually works
	Cache reference rate
	perf stat -e cache-references
	Validates L2 tiling strategy
	GPU occupancy %
	nvprof / Nsight Compute
	Kernel efficiency on device
	Memory bandwidth (GB/s)
	Nsight Systems timeline
	PCIe and global mem utilization
	Sortedness vs speedup curve
	Custom benchmark script
	Workload characterization figure
	Prediction accuracy %
	Feedback log analysis
	Validates cost model correctness
	

8.4  The Key Figure to Produce
The most impactful benchmark output is a 2D scatter plot: X-axis = sortedness score from profiler (0.0 to 1.0), Y-axis = PASE speedup vs std::sort. Color points by chosen strategy. This single figure tells the complete story of when PASE wins and why.


Target claim for README / resume
	PASE achieves 2.1x speedup over std::sort on nearly-sorted inputs (sortedness > 0.85),
	3.4x speedup over Python Timsort on heavy-duplicate workloads (dup_ratio > 0.45),
	and 1.8x speedup over std::sort on large random arrays (n > 1M) via GPU dispatch.
	

	

________________


9. Project Structure




pase/
  include/
    pase.h               # public API: adaptive_sort(), Profile, Strategy
    profiler.h           # Profiler class
    cost_model.h         # CostModel class
    dispatcher.h         # Dispatcher
    feedback.h           # FeedbackLogger, ThresholdTuner
  src/
    profiler.cpp         # sampling profiler implementation
    cost_model.cpp       # GPU and CPU cost estimators
    dispatcher.cpp       # strategy selector
    cpu/
      insertion.cpp      # binary-search insertion sort
      run_merge.cpp      # run detection + extension + cache-tiled merge
      quicksort_3way.cpp # fat-partition 3-way quicksort
      introsort.cpp      # introsort (QS + heapsort fallback)
    gpu/
      block_sort.cu      # bitonic sort kernel (shared memory)
      merge_pass.cu      # parallel merge kernel
      gpu_sort.cu        # orchestration: streams, async transfers
    feedback.cpp         # SortLog writer, EMA tuner
    pase.cpp             # adaptive_sort() entry point
  bench/
    bench_main.cpp       # benchmark harness (all datasets)
    gen_datasets.cpp     # dataset generators
    compare_std.cpp      # comparison vs std::sort, stable_sort
    compare_thrust.cu    # comparison vs thrust::sort
    plot_results.py      # generate sortedness-vs-speedup figure
  tune/
    grid_search.py       # offline threshold optimizer
    analyze_log.py       # feedback log analysis + prediction accuracy
  tests/
    test_correctness.cpp # correctness on all datasets (gtest)
    test_profiler.cpp    # profiler metric validation
  CMakeLists.txt
  README.md


________________


10. Implementation Phases




Phases are ordered by dependency, not by difficulty. Phase 1 is the foundation — do not skip it or rush it. Every later phase depends on the profiler and dispatcher being correct.


Phase 1
Days 1–3
	Phase 2
Days 4–7
	Phase 3
Days 8–14
	Phase 4
Days 15–21
	

Phase 1 — Foundation (Days 1–3)
Deliverables
	- Sampling profiler: all 6 metrics, parameterized sample_rate, --verbose output
	- CPU dispatcher: rule-based (no cost model yet), routes to 4 CPU strategies
	- Correctness tests: gtest suite passing on all 10 benchmark datasets
	- Baseline benchmark: PASE vs std::sort, first numbers on board
	

	

* Implement Profile struct and Profiler::analyze() — single-pass sampler
* Implement rule-based dispatcher (no cost model yet — add in Phase 2)
* Implement INSERTION_OPT — binary search + memmove version
* Implement INTROSORT — std::sort wrapper is acceptable for Phase 1
* Write gtest correctness suite: sorted, reverse, random, nearly_sorted
* Run first benchmark: PASE vs std::sort on all datasets, record numbers


Phase 2 — CPU Engine + Cost Model (Days 4–7)
Deliverables
	- Full CPU engine: RUN_MERGE_OPT with run extension, cache-tiled merge
	- THREE_WAY_QS with fat-partition
	- Cost model: GPU and CPU estimators, PCIe model, 0.85 margin
	- CPU throughput calibration at startup
	

	

* Implement run_detection() and extend_run() with repair_budget
* Implement cache_tiled_merge() with L2-detected tile size
* Implement quicksort_3way() fat-partition
* Implement CostModel::estimate_gpu() and estimate_cpu()
* Replace rule-based dispatch with cost-model dispatch
* Add CPU throughput calibration (sort known workload at startup, record ops/ms)
* Benchmark: cache miss comparison with perf stat — verify cache improvement


Phase 3 — CUDA Engine + Feedback Loop (Days 8–14)
Deliverables
	- GPU engine: bitonic block sort + parallel merge passes
	- CUDA stream overlap for async H2D/D2H
	- FeedbackLogger writing to CSV after every sort
	- EMA threshold tuner running online
	- Optional: CPU-GPU concurrent sort (Phase 3B stretch goal)
	

	

* Implement blockBitonicSort kernel — bitonic network in shared memory
* Implement parallelMerge kernel — coalesced global memory, one pass per launch
* Implement gpu_sort() orchestration: streams, async transfers, padding
* Use cudaOccupancyMaxPotentialBlockSize for auto block-size tuning
* Profile with Nsight Compute: verify occupancy > 70%, coalesced access
* Implement FeedbackLogger: SortLog struct, CSV writer
* Implement EMA threshold updater: update after each sort
* Benchmark: PASE GPU vs thrust::sort on large_random_1M and 10M


Phase 4 — Optimization, Tuning & Report (Days 15–21)
Deliverables
	- Offline grid search optimizer over collected feedback log
	- Sortedness-vs-speedup figure (the key benchmark chart)
	- Full README with architecture diagram, benchmark results, usage examples
	- Optional: replace grid search with decision tree (scikit-learn)
	

	

* Run full benchmark suite: all 10 datasets x all comparison targets
* Run grid_search.py on accumulated feedback log; apply optimized thresholds
* Generate sortedness-vs-speedup scatter plot with plot_results.py
* Measure prediction accuracy of cost model from feedback log
* Write README: motivation, architecture diagram, benchmark figures, results table
* [Stretch] Train sklearn DecisionTreeClassifier on log features; compare to grid search
* [Stretch] Implement CPU-GPU concurrent sort for very large arrays


________________


11. Risk Register




Risk
	Severity
	Mitigation
	Fallback
	GPU not available (no CUDA device)
	Medium
	Develop CPU path first; mock GPU dispatch
	Ship CPU-only build; document GPU as optional
	PCIe cost model inaccurate
	Medium
	Calibrate H2D bandwidth at startup empirically
	Widen GPU margin from 0.85 to 0.70
	Profiler overhead too high for small n
	Low
	Skip profiler for n < 1000; use INSERTION_OPT directly
	Hard-code small-n path in dispatcher
	PASE slower than std::sort on random
	Low
	INTROSORT fallback wraps std::sort; parity guaranteed
	Document as known limitation; out-of-scope
	Feedback log grows too large
	Low
	Cap log at 10k entries; rotate/truncate oldest
	Disable logging by default; enable via flag
	

________________


12. Success Criteria




Minimum Viable (A project)
* Profiler correctly classifies all 10 benchmark datasets
* CPU engine outperforms std::sort on nearly_sorted and heavy_dup datasets
* CUDA engine correctly sorts large_random_1M and 10M
* Feedback logger writes valid CSV; EMA tuner runs without error
* Benchmark suite produces timing comparisons on all datasets
* Correctness tests pass 100% on all datasets


Target (A+ project / interview-defining)
* Cost model prediction accuracy > 80% on held-out benchmark runs
* Cache miss reduction > 30% vs naive mergesort on arrays > 4MB (perf stat)
* GPU speedup > 1.5x vs thrust::sort on 10M random ints
* Sortedness-vs-speedup figure shows clean monotonic relationship
* --verbose mode explains every dispatch decision correctly
* Grid search produces measurably better thresholds than hand-tuned defaults


Stretch (top 1% portfolio piece)
* CPU-GPU concurrent sort operational and benchmarked
* Decision tree classifier replaces grid search; accuracy > 85%
* Blog post or write-up documenting system design and benchmark findings
* Open source release with clean README; cited by others






PASE — Predictive Adaptive Sorting Engine
Product Development Document v1.0 — MindCore LLC — March 2026
MindCore LLC — ConfidentialPage
