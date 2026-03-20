#include "runtime.h"

#include "config_loader.h"
#include "threshold_tuner.h"

#include <mutex>

namespace pase {

namespace {

struct LoadedPaseConfig {
  Dispatcher::Thresholds th;
  double gpu_win = 0.85;
  CostModelFit fit;
};

LoadedPaseConfig& loaded_pase_config() {
  static LoadedPaseConfig cfg;
  static std::once_flag once;
  std::call_once(once, [] {
    cfg.gpu_win = global_threshold_tuner().gpu_win_factor();
    load_pase_config_file(default_pase_config_path(), cfg.th, cfg.gpu_win,
                         &cfg.fit);
    global_threshold_tuner().set_gpu_win_factor(cfg.gpu_win);
  });
  return cfg;
}

Dispatcher make_dispatcher_from_config() {
  return Dispatcher(loaded_pase_config().th);
}

}  // namespace

CostModel& global_cost_model() {
  static CostModel model;
  static std::once_flag once;
  std::call_once(once, [] {
    CostModel::calibrate_with_int_sort(model);
    model.apply_fit(loaded_pase_config().fit);
  });
  return model;
}

const Dispatcher& runtime_dispatcher() {
  static Dispatcher d = make_dispatcher_from_config();
  return d;
}

}  // namespace pase
