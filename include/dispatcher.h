#pragma once

#include "profiler.h"
#include "strategies.h"

namespace pase {

/**
 * Dispatcher: routes to optimal strategy based on Profile.
 * Phase 1: rule-based (no cost model).
 * Phase 2+: cost-model-driven.
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

  Strategy select_strategy(const Profile& p) const;

 private:
  Thresholds thresholds_;
};

}  // namespace pase
