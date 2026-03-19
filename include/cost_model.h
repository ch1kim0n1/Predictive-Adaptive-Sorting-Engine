#pragma once

#include "profiler.h"
#include "strategies.h"

#include <cmath>
#include <cstddef>

namespace pase {

/**
 * Calibrated cost model: GPU vs CPU estimates (PDD §4).
 * cpu_ops_per_ms is fit from a timed int std::sort calibration run.
 */
class CostModel {
 public:
  CostModel();

  void set_cpu_ops_per_ms(double v) { cpu_ops_per_ms_ = v; }
  double cpu_ops_per_ms() const { return cpu_ops_per_ms_; }

  /// Wall-clock ms estimate for GPU path (PCIe + kernel), Phase 3+.
  double estimate_gpu(int n, float entropy, std::size_t elem_size) const;

  /// Wall-clock ms estimate for a CPU strategy.
  double estimate_cpu(const Profile& p, Strategy s) const;

  /**
   * Pick cheapest CPU strategy among INTROSORT and heuristically eligible
   * RUN_MERGE / THREE_WAY_QS / INSERTION (when sortedness is high but below
   * dispatcher's INSERTION fast-path threshold).
   */
  Strategy best_cpu_strategy(const Profile& p, float sorted_insertion_thr,
                             int run_merge_thr, float dup_thr) const;

  /** One-time calibration using std::sort(random int[N]). */
  static void calibrate_with_int_sort(CostModel& out);

  /// Default margin: GPU must beat CPU by this factor to win.
  static constexpr double kGpuMargin = 0.85;

 private:
  double cpu_ops_per_ms_;
  static constexpr double kPcieGbPerS = 12.0;
  static constexpr double kGpuTflops = 5.0;
};

}  // namespace pase
