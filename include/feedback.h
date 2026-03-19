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

class FeedbackLogger {
 public:
  void log(const SortLog& entry);
  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool enabled() const { return enabled_; }

 private:
  bool enabled_ = false;
};

/** Logger used by adaptive_sort when PASE_FEEDBACK=1 or set_feedback_logging(true). */
FeedbackLogger& global_feedback_logger();

void set_feedback_logging(bool enabled);
bool feedback_logging_enabled();

}  // namespace pase
