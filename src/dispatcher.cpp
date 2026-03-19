#include "dispatcher.h"

namespace pase {

Dispatcher::Dispatcher(const Thresholds& thr) : thresholds_(thr) {}

Strategy Dispatcher::select_strategy(const Profile& p, const CostModel& cm,
                                    std::size_t element_size, bool gpu_available,
                                    double gpu_win_factor) const {
  if (p.sortedness > thresholds_.sorted) {
    return Strategy::INSERTION_OPT;
  }

  Strategy best_cpu =
      cm.best_cpu_strategy(p, thresholds_.sorted, thresholds_.run_merge,
                          thresholds_.dup);
  double cpu_ms = cm.estimate_cpu(p, best_cpu);
  double gpu_ms = cm.estimate_gpu(p.n, p.entropy, element_size);

  if (gpu_available && p.n >= thresholds_.min_gpu &&
      gpu_ms < cpu_ms * gpu_win_factor) {
    return Strategy::GPU_SORT;
  }

  return best_cpu;
}

}  // namespace pase
