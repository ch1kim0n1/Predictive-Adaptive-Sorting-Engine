#pragma once

#include "cost_model_fit.h"
#include "dispatcher.h"

#include <string>

namespace pase {

/**
 * Load ~/.pase/optimized_thresholds.json unless PASE_CONFIG overrides path.
 * File must be valid JSON. Partial keys ok: only present fields override defaults.
 * @param gpu_win_factor in/out default 0.85; updated if key present
 * @return true if file opened and at least one key parsed
 */
bool load_pase_config(Dispatcher::Thresholds& th, double& gpu_win_factor);

/**
 * @param cost_fit optional; if non-null, fills cost_fit from "cost_fit" JSON object.
 * @return true if file opened and at least one key parsed (thresholds, gpu_win, or cost_fit)
 */
bool load_pase_config_file(const std::string& path, Dispatcher::Thresholds& th,
                          double& gpu_win_factor,
                          CostModelFit* cost_fit = nullptr);

std::string default_pase_config_path();

}  // namespace pase
