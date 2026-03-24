#include "dispatcher.h"

namespace pase {

Dispatcher::Dispatcher(const Thresholds& thr) : thresholds_(thr) {}

Strategy Dispatcher::select_strategy(const Profile& p, const CostModel& cm,
                                    std::size_t element_size, bool gpu_available,
                                    double gpu_win_factor) const {
  auto insertion_cap_for = [&](const Profile& prof) -> int {
    int cap = thresholds_.max_insertion_n;
    if (prof.sortedness >= 0.98f && prof.entropy <= 0.35f) {
      cap *= 4;
    } else if (prof.sortedness >= 0.95f && prof.entropy <= 0.50f) {
      cap *= 2;
    }
    return cap;
  };

  auto apply_guardrail = [&](Strategy s) -> Strategy {
    if (s == Strategy::GPU_SORT) {
      return s;
    }
    const double intro_ms = cm.estimate_cpu(p, Strategy::INTROSORT);
    const double est_ms = cm.estimate_cpu(p, s);
    if (est_ms >
        static_cast<double>(thresholds_.strategy_guardrail) * intro_ms) {
      return Strategy::INTROSORT;
    }
    return s;
  };

  const int insertion_cap = insertion_cap_for(p);
  if (p.n <= insertion_cap &&
      p.sortedness > thresholds_.sorted) {
    return Strategy::INSERTION_OPT;
  }

  Strategy best_cpu =
      cm.best_cpu_strategy(p, thresholds_.sorted, thresholds_.run_merge,
                          thresholds_.dup, insertion_cap);

  if (best_cpu == Strategy::THREE_WAY_QS) {
    const double intro_ms = cm.estimate_cpu(p, Strategy::INTROSORT);
    const double three_ms = cm.estimate_cpu(p, Strategy::THREE_WAY_QS);
    const bool dup_extreme = p.duplicate_ratio >= 0.995f;
    const bool large_enough = p.n >= 100000;
    const bool clear_win = three_ms < intro_ms * 0.95;
    if (!(dup_extreme && large_enough && clear_win)) {
      best_cpu = Strategy::INTROSORT;
    }
  }

  auto conservative_near_border = [&](Strategy s) -> Strategy {
    if (s == Strategy::INTROSORT || s == Strategy::INSERTION_OPT ||
        s == Strategy::GPU_SORT) {
      return s;
    }
    const double intro_ms = cm.estimate_cpu(p, Strategy::INTROSORT);
    const double spec_ms = cm.estimate_cpu(p, s);
    const double need =
        static_cast<double>(thresholds_.conservative_specialist_frac);
    if (s == Strategy::THREE_WAY_QS) {
      const float lo = thresholds_.dup - thresholds_.dup_border_band;
      const float hi = thresholds_.dup + thresholds_.dup_border_band;
      if (p.duplicate_ratio >= lo && p.duplicate_ratio <= hi &&
          !(spec_ms < intro_ms * need)) {
        return Strategy::INTROSORT;
      }
    }
    if (s == Strategy::RUN_MERGE_OPT) {
      const int lo = thresholds_.run_merge - thresholds_.run_merge_border;
      const int hi = thresholds_.run_merge + thresholds_.run_merge_border;
      if (p.avg_run_length >= lo && p.avg_run_length <= hi &&
          !(spec_ms < intro_ms * need)) {
        return Strategy::INTROSORT;
      }
    }
    return s;
  };
  best_cpu = conservative_near_border(best_cpu);
  best_cpu = apply_guardrail(best_cpu);

  const double cpu_ms = cm.estimate_cpu(p, best_cpu);
  const double gpu_ms =
      cm.estimate_gpu(p.n, p.entropy, element_size);

  if (gpu_available && p.n >= thresholds_.min_gpu &&
      gpu_ms * thresholds_.gpu_rel_margin < cpu_ms * gpu_win_factor) {
    return Strategy::GPU_SORT;
  }

  return best_cpu;
}

}  // namespace pase
