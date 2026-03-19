#pragma once

namespace pase {

/**
 * Strategy enum: which sorting algorithm to use.
 * - INSERTION_OPT: For nearly-sorted data (sortedness > 0.90)
 * - RUN_MERGE_OPT: For structured data with long ascending runs
 * - THREE_WAY_QS: For heavy duplicates (dup_ratio > 0.40)
 * - INTROSORT: Fallback for random data
 * - GPU_SORT: GPU dispatch (Phase 3)
 */
enum class Strategy {
  INSERTION_OPT,
  RUN_MERGE_OPT,
  THREE_WAY_QS,
  INTROSORT,
  GPU_SORT
};

}  // namespace pase
