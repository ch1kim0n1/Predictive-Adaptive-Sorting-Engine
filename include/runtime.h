#pragma once

#include "cost_model.h"
#include "dispatcher.h"

namespace pase {

/** Calibrated singleton cost model (Phase 2+). */
CostModel& global_cost_model();

/**
 * Dispatcher built from defaults + optional ~/.pase/optimized_thresholds.json
 * (or PASE_CONFIG path). Applies gpu_win_factor to ThresholdTuner when present.
 */
const Dispatcher& runtime_dispatcher();

}  // namespace pase
