#pragma once

#include "cost_model.h"
#include "profiler.h"
#include "strategies.h"

#include <cstddef>

namespace pase {

/**
 * Dispatcher: routes using fast heuristics + cost model (Phase 2).
 */
class Dispatcher {
 public:
  struct Thresholds {
    /** Sortedness above which insertion may be chosen (only if n <= max_insertion_n). */
    float sorted;
    int run_merge;
    float dup;
    int min_gpu;
    /** Insertion sort only for n <= this; avoids O(n²) pathologies on large arrays. */
    int max_insertion_n;
    /**
     * If model estimates chosen CPU strategy slower than this × introsort estimate,
     * fall back to INTROSORT.
     */
    float strategy_guardrail;
    /** GPU picked only if gpu_ms * gpu_rel_margin < cpu_ms * gpu_win_factor. */
    float gpu_rel_margin;
    /**
     * Near dup threshold: THREE_WAY only if model est_three_way < est_intro * this
     * fraction (else INTROSORT). Set to 1 to disable border conservative behavior.
     */
    float dup_border_band;
    int run_merge_border;
    /** RUN_MERGE border ± run_merge_border avg_run_length uses same rule as dup. */
    float conservative_specialist_frac;
    Thresholds()
        : sorted(0.90f),
          run_merge(20),
          dup(0.90f),
          min_gpu(250000),
          max_insertion_n(1024),
          strategy_guardrail(2.25f),
          gpu_rel_margin(1.12f),
          dup_border_band(0.08f),
          run_merge_border(6),
          conservative_specialist_frac(0.96f) {}
  };

  explicit Dispatcher(const Thresholds& thr = Thresholds());

  const Thresholds& thresholds() const { return thresholds_; }

  /**
   * INSERTION fast-path, then GPU vs best CPU if gpu_available, else best CPU.
   * gpu_win_factor: GPU wins if gpu_est < cpu_est * factor (default ~0.85).
   */
  Strategy select_strategy(const Profile& p, const CostModel& cm,
                          std::size_t element_size, bool gpu_available,
                          double gpu_win_factor) const;

 private:
  Thresholds thresholds_;
};

}  // namespace pase
