#include "strategies.h"

namespace pase {

const char* strategy_name(Strategy s) {
  switch (s) {
    case Strategy::INSERTION_OPT:
      return "INSERTION_OPT";
    case Strategy::RUN_MERGE_OPT:
      return "RUN_MERGE_OPT";
    case Strategy::THREE_WAY_QS:
      return "THREE_WAY_QS";
    case Strategy::INTROSORT:
      return "INTROSORT";
    case Strategy::GPU_SORT:
      return "GPU_SORT";
  }
  return "UNKNOWN";
}

}  // namespace pase
