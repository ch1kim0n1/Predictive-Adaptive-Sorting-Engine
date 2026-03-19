#include "cost_model.h"

namespace pase {

double CostModel::estimate_gpu(int /*n*/, float /*entropy*/) const {
  return 1e9;  // Stub: always expensive (no GPU in Phase 1)
}

double CostModel::estimate_cpu(const Profile& /*p*/, Strategy /*s*/) const {
  return 0.0;  // Stub
}

Strategy CostModel::best_cpu_strategy(const Profile& /*p*/) const {
  return Strategy::INTROSORT;  // Stub
}

}  // namespace pase
