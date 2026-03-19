#pragma once

#include "strategies.h"

#include <mutex>

namespace pase {

/**
 * Online EMA-style tuning of GPU dispatch margin (Phase 3).
 * GPU wins when: gpu_est_ms < cpu_est_ms * gpu_win_factor().
 * Default factor 0.85 (PDD penalty). Nudged when feedback logging observes error.
 */
class ThresholdTuner {
 public:
  ThresholdTuner();

  double gpu_win_factor() const;

  /**
   * @param predicted_gpu_ms cost model GPU estimate
   * @param actual_ms measured wall time for chosen path
   * @param chosen dispatched strategy
   */
  void observe_gpu_decision(double predicted_gpu_ms, double actual_ms,
                           Strategy chosen);

 private:
  mutable std::mutex mu_;
  double gpu_win_factor_;
  static constexpr double kAlpha = 0.02;
  static constexpr double kMinFactor = 0.55;
  static constexpr double kMaxFactor = 0.95;
};

ThresholdTuner& global_threshold_tuner();

}  // namespace pase
