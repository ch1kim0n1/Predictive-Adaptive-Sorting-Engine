#include "config_loader.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pase {

namespace {

void parse_cost_fit(const nlohmann::json& cf, CostModelFit* out, bool* any) {
  if (!out || !cf.is_object()) {
    return;
  }
  auto take_d = [&](const char* key, double& slot) {
    if (cf.contains(key) && cf[key].is_number()) {
      slot = cf[key].get<double>();
      *any = true;
    }
  };
  take_d("introsort", out->scale_introsort);
  take_d("run_merge", out->scale_run_merge);
  take_d("three_way", out->scale_three_way);
  take_d("insertion", out->scale_insertion);
  take_d("gpu_kernel", out->gpu_kernel_scale);
  take_d("profile_bias_mult", out->profile_bias_mult);
}

}  // namespace

std::string default_pase_config_path() {
  const char* env = std::getenv("PASE_CONFIG");
  if (env && env[0]) {
    return std::string(env);
  }
  const char* home = std::getenv("HOME");
  if (!home) {
    return {};
  }
  return std::string(home) + "/.pase/optimized_thresholds.json";
}

bool load_pase_config_file(const std::string& path, Dispatcher::Thresholds& th,
                          double& gpu_win_factor, CostModelFit* cost_fit) {
  if (path.empty()) {
    return false;
  }
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  if (text.empty()) {
    return false;
  }

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::parse_error&) {
    return false;
  }

  if (!j.is_object()) {
    return false;
  }

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
  if (j.contains("gpu_win_factor") && j["gpu_win_factor"].is_number()) {
    gpu_win_factor = j["gpu_win_factor"].get<double>();
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
  if (j.contains("cost_fit") && j["cost_fit"].is_object()) {
    any = true;
    if (cost_fit != nullptr) {
      parse_cost_fit(j["cost_fit"], cost_fit, &any);
    }
  }

  return any;
}

bool load_pase_config(Dispatcher::Thresholds& th, double& gpu_win_factor) {
  return load_pase_config_file(default_pase_config_path(), th, gpu_win_factor,
                              nullptr);
}

}  // namespace pase
