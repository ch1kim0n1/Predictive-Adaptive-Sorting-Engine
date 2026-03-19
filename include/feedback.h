#pragma once

#include "strategies.h"

namespace pase {

/**
 * SortLog: single entry in feedback log.
 * Written after every sort for offline/online tuning.
 */
struct SortLog {
  float sortedness;
  float duplicate_ratio;
  float entropy;
  int avg_run_length;
  int n;
  Strategy chosen_strategy;
  double predicted_cpu_ms;
  double predicted_gpu_ms;
  double actual_ms;
  bool prediction_correct;
};

/**
 * FeedbackLogger: writes SortLog to CSV after each sort.
 * Phase 2: logging only. Phase 3: EMA tuner.
 */
class FeedbackLogger {
 public:
  void log(const SortLog& entry);
  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool enabled() const { return enabled_; }

 private:
  bool enabled_ = false;
};

}  // namespace pase
