#pragma once

#include "profiler.h"
#include "strategies.h"

#include <cstddef>

namespace pase {

/** Preview dispatcher + cost model for a profile (no sorting). */
struct DispatchPreview {
  Strategy strategy;
  Strategy best_cpu;
  double pred_gpu_ms;
  double pred_cpu_ms;
};

DispatchPreview preview_dispatch_for_profile(const Profile& p,
                                            std::size_t elem_size,
                                            bool gpu_available);

}  // namespace pase
