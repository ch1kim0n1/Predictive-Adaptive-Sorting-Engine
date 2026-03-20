#pragma once

#include <complex>

namespace pase {

/** Lexicographic order on (real, imag); use with `adaptive_sort` for GPU complex path. */
template <typename T>
struct LexicographicComplexLess {
  bool operator()(const std::complex<T>& a, const std::complex<T>& b) const {
    const T ar = a.real(), br = b.real();
    if (ar < br) {
      return true;
    }
    if (br < ar) {
      return false;
    }
    return a.imag() < b.imag();
  }
};

}  // namespace pase
