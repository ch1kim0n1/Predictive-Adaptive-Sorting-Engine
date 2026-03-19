#pragma once

#include "profiler.h"
#include "strategies.h"

namespace pase {

/**
 * CostModel: estimates expected runtime for each strategy.
 * Phase 2: full implementation with GPU/CPU estimators.
 * Phase 1: stubs only.
 */
class CostModel {
 public:
  /// Estimate GPU cost (PCIe transfer + kernel time). Phase 2+.
  double estimate_gpu(int n, float entropy) const;

  /// Estimate CPU cost for given strategy. Phase 2+.
  double estimate_cpu(const Profile& p, Strategy s) const;

  /// Best CPU strategy for profile. Phase 2+.
  Strategy best_cpu_strategy(const Profile& p) const;
};

}  // namespace pase
