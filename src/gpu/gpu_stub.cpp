#include "gpu_api.h"

namespace pase {

bool gpu_sort_int_available() { return false; }

bool gpu_sort_int(int* /*host_data*/, int /*n*/) { return false; }

}  // namespace pase
