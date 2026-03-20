#include "threshold_tuner.h"

#include <algorithm>
#include <cmath>

namespace pase {

ThresholdTuner::ThresholdTuner() : gpu_win_factor_(0.85) {}

double ThresholdTuner::gpu_win_factor() const {
  std::lock_guard<std::mutex> lock(mu_);
  return gpu_win_factor_;
}

void ThresholdTuner::set_gpu_win_factor(double v) {
  std::lock_guard<std::mutex> lock(mu_);
  gpu_win_factor_ = std::clamp(v, kMinFactor, kMaxFactor);
}

void ThresholdTuner::observe_gpu_decision(double predicted_gpu_ms,
                                         double actual_ms, Strategy chosen) {
  if (chosen != Strategy::GPU_SORT) {
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  const double pred = std::max(predicted_gpu_ms, 1e-9);
  const double rel_err = (actual_ms - predicted_gpu_ms) / pred;
  if (rel_err > 0.15) {
    gpu_win_factor_ -= kAlpha * rel_err;
  } else if (rel_err < -0.15) {
    gpu_win_factor_ += kAlpha * (-rel_err);
  }
  gpu_win_factor_ = std::clamp(gpu_win_factor_, kMinFactor, kMaxFactor);
}

ThresholdTuner& global_threshold_tuner() {
  static ThresholdTuner t;
  return t;
}

}  // namespace pase
