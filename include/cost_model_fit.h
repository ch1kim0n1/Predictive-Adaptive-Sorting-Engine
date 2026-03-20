#pragma once

namespace pase {

/**
 * Optional multipliers loaded from JSON ("cost_fit" object) to nudge the cost
 * model toward observed hardware after offline fitting (see tune/fit_cost_model.py).
 * All default to 1.0 (no effect).
 */
struct CostModelFit {
  double scale_introsort = 1.0;
  double scale_run_merge = 1.0;
  double scale_three_way = 1.0;
  double scale_insertion = 1.0;
  /// Scales the compute (kernel) term in estimate_gpu; transfer kept separate.
  double gpu_kernel_scale = 1.0;
  double profile_bias_mult = 1.0;
};

}  // namespace pase
