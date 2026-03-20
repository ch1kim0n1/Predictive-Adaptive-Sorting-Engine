#include "gpu_api.h"

#include <complex>

namespace pase {

bool gpu_sort_int_available() { return false; }

bool gpu_sort_device_available() { return false; }

bool gpu_sort_int(int* /*host_data*/, int /*n*/) { return false; }

bool gpu_sort_float(float* /*host_data*/, int /*n*/) { return false; }

bool gpu_sort_double(double* /*host_data*/, int /*n*/) { return false; }

bool gpu_sort_complex_float(std::complex<float>* /*host_data*/, int /*n*/) {
  return false;
}

bool gpu_sort_complex_double(std::complex<double>* /*host_data*/, int /*n*/) {
  return false;
}

}  // namespace pase
