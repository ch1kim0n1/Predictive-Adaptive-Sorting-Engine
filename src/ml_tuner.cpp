#include "ml_tuner.h"

#if defined(PASE_WITH_ML_TUNING)

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace pase {
namespace ml_tuning {

namespace {

bool merge_json_into_thresholds(const nlohmann::json& j, Dispatcher::Thresholds& th) {
  bool any = false;
  if (j.contains("sorted") && j["sorted"].is_number()) {
    th.sorted = j["sorted"].get<float>();
    any = true;
  }
  if (j.contains("run_merge") && j["run_merge"].is_number()) {
    th.run_merge = static_cast<int>(j["run_merge"].get<double>());
    any = true;
  }
  if (j.contains("dup") && j["dup"].is_number()) {
    th.dup = j["dup"].get<float>();
    any = true;
  }
  if (j.contains("min_gpu") && j["min_gpu"].is_number()) {
    th.min_gpu = static_cast<int>(j["min_gpu"].get<double>());
    any = true;
  }
  if (j.contains("max_insertion_n") && j["max_insertion_n"].is_number()) {
    th.max_insertion_n = static_cast<int>(j["max_insertion_n"].get<double>());
    any = true;
  }
  if (j.contains("strategy_guardrail") && j["strategy_guardrail"].is_number()) {
    th.strategy_guardrail = j["strategy_guardrail"].get<float>();
    any = true;
  }
  if (j.contains("gpu_rel_margin") && j["gpu_rel_margin"].is_number()) {
    th.gpu_rel_margin = j["gpu_rel_margin"].get<float>();
    any = true;
  }
  if (j.contains("dup_border_band") && j["dup_border_band"].is_number()) {
    th.dup_border_band = j["dup_border_band"].get<float>();
    any = true;
  }
  if (j.contains("run_merge_border") && j["run_merge_border"].is_number()) {
    th.run_merge_border = static_cast<int>(j["run_merge_border"].get<double>());
    any = true;
  }
  if (j.contains("conservative_specialist_frac") &&
      j["conservative_specialist_frac"].is_number()) {
    th.conservative_specialist_frac =
        j["conservative_specialist_frac"].get<float>();
    any = true;
  }
  return any;
}

}  // namespace

void apply_ml_threshold_file(Dispatcher::Thresholds& th) {
  const char* env = std::getenv("PASE_ML_CONFIG");
  fs::path path;
  if (env && env[0]) {
    path = env;
  } else {
    const char* home = std::getenv("HOME");
    if (!home) {
      return;
    }
    path = fs::path(home) / ".pase" / "ml_thresholds.json";
  }
  if (!fs::exists(path)) {
    return;
  }
  std::ifstream in(path);
  if (!in) {
    return;
  }
  try {
    nlohmann::json j;
    in >> j;
    merge_json_into_thresholds(j, th);
  } catch (...) {
    /* Ignore bad ML JSON; keep base thresholds. */
  }
}

}  // namespace ml_tuning
}  // namespace pase

#else

namespace pase {
namespace ml_tuning {

void apply_ml_threshold_file(Dispatcher::Thresholds&) {}

}  // namespace ml_tuning
}  // namespace pase

#endif
