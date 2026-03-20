#pragma once

#include <cstddef>
#include <vector>

#if defined(PASE_WITH_MPI)
#include <mpi.h>
#endif

namespace pase {

/**
 * Single-node convenience: adaptive sort of local buffer (Phase 5 baseline).
 */
void distributed_sort_local_int(int* buf, int n);

#if defined(PASE_WITH_MPI)
/**
 * MPI collective **global** sort for `int`.
 *
 * - Gathers all elements on rank 0, sorts with `adaptive_sort`, then scatters
 *   contiguous **balanced** slices so each rank receives
 *   floor(N/size) or ceil(N/size) keys.
 * - On return, `local_chunk` is **resized** to the new slice length (may differ
 *   from input).
 * - If the **gathered** buffer would exceed `kMaxGatherBytes` (256 MiB of int
 *   payload), falls back to **local-only** `adaptive_sort` on each rank (no
 *   global order).
 *
 * @param global_n_hint If >= 0, must equal the MPI sum of local sizes; ignored
 *                      on mismatch (recomputed via Allreduce).
 */
void distributed_sort_mpi_int(MPI_Comm comm, std::vector<int>& local_chunk,
                             long long global_n_hint = -1);

/** Gather cap used by `distributed_sort_mpi_int` (payload bytes on rank 0). */
constexpr std::size_t distributed_sort_mpi_gather_cap_bytes() noexcept {
  return 256ull * 1024 * 1024;
}
#endif

}  // namespace pase
