#include "pase.h"

#include <vector>

namespace pase {

template void adaptive_sort<int, std::less<int>>(int*, int, const std::less<int>&,
                                                bool);
template void adaptive_sort<int, std::less<int>>(std::vector<int>&,
                                                  const std::less<int>&, bool);

}  // namespace pase
