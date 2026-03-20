#pragma once

/**
 * Versioned evaluation contract for bench_results and regression tests.
 * Bump when dataset types, default sizes, or acceptance rules change.
 */
#define PASE_BENCH_SUITE_VERSION "1.2"

namespace pase::bench_contract {

/** Documented targets: correctness always; perf vs std::sort on reference hardware. */
constexpr double kAcceptStructuredMaxSlowdown = 1.65;  // nearly_sorted_95 @ 100k
/** Fully sorted @ 100k: sampling + cost model can dominate vs libc `std::sort` on small timings. */
constexpr double kAcceptFullySortedMaxSlowdown = 4.0;
constexpr double kAcceptRandomMaxSlowdown = 2.0;  // loose bound; CI reference

}  // namespace pase::bench_contract
