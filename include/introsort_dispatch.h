#pragma once

namespace pase {
namespace cpu {

/** CPU introsort entry for int / std::less — may use SIMD-assisted paths when built with them. */
void pase_cpu_introsort_int_dispatch(int* array, int n);

}  // namespace cpu
}  // namespace pase
