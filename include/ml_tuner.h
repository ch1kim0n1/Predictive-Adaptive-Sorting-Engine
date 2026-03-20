#pragma once

#include "dispatcher.h"

namespace pase {
namespace ml_tuning {

/**
 * Merge optional ML-generated threshold overrides from JSON.
 * Path: PASE_ML_CONFIG env, else ~/.pase/ml_thresholds.json if it exists.
 * Keys match dispatcher config (sorted, run_merge, dup, …). Ignored if file
 * missing or PASE built without PASE_WITH_ML_TUNING.
 */
void apply_ml_threshold_file(Dispatcher::Thresholds& th);

}  // namespace ml_tuning
}  // namespace pase
