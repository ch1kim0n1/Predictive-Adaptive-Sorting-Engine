#include "dispatcher.h"

namespace pase {

Dispatcher::Dispatcher(const Thresholds& thr) : thresholds_(thr) {}

Strategy Dispatcher::select_strategy(const Profile& p) const {
  if (p.sortedness > thresholds_.sorted) {
    return Strategy::INSERTION_OPT;
  }
  if (p.avg_run_length > thresholds_.run_merge) {
    return Strategy::RUN_MERGE_OPT;
  }
  if (p.duplicate_ratio > thresholds_.dup) {
    return Strategy::THREE_WAY_QS;
  }
  return Strategy::INTROSORT;
}

}  // namespace pase
