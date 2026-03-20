#include "pase_plan.h"

#include "cost_model.h"
#include "dispatcher.h"
#include "runtime.h"
#include "threshold_tuner.h"

namespace pase {

DispatchPreview preview_dispatch_for_profile(const Profile& p,
                                            std::size_t elem_size,
                                            bool gpu_available) {
  CostModel& cm = global_cost_model();
  const Dispatcher& d = runtime_dispatcher();
  const double win = global_threshold_tuner().gpu_win_factor();
  Strategy best_cpu = cm.best_cpu_strategy(
      p, d.thresholds().sorted, d.thresholds().run_merge, d.thresholds().dup,
      d.thresholds().max_insertion_n);
  Strategy s =
      d.select_strategy(p, cm, elem_size, gpu_available, win);
  const double pred_gpu = cm.estimate_gpu(p.n, p.entropy, elem_size);
  double pred_cpu = cm.estimate_cpu(p, s);
  if (s == Strategy::GPU_SORT) {
    pred_cpu = cm.estimate_cpu(p, best_cpu);
  }
  return DispatchPreview{s, best_cpu, pred_gpu, pred_cpu};
}

}  // namespace pase
