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
    float sorted;
    int run_merge;
    float dup;
    int min_gpu;
    Thresholds()
        : sorted(0.90f), run_merge(64), dup(0.40f), min_gpu(100000) {}
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
