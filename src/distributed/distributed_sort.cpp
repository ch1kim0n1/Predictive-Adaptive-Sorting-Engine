#include "distributed_sort.h"

#include <pase.h>

#include <cstdio>

#if defined(PASE_WITH_MPI)
#include <mpi.h>
#endif

namespace pase {

void distributed_sort_local_int(int* buf, int n) {
  adaptive_sort(buf, n);
}

#if defined(PASE_WITH_MPI)

namespace {

void sort_locally(std::vector<int>& local_chunk) {
  const int n = static_cast<int>(local_chunk.size());
  if (n > 0) {
    adaptive_sort(local_chunk.data(), n);
  }
}

}  // namespace

void distributed_sort_mpi_int(MPI_Comm comm, std::vector<int>& local_chunk,
                             long long global_n_hint) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const int local_n = static_cast<int>(local_chunk.size());
  long long sum_n = local_n;
  MPI_Allreduce(MPI_IN_PLACE, &sum_n, 1, MPI_LONG_LONG, MPI_SUM, comm);

  if (global_n_hint >= 0 && sum_n != global_n_hint && rank == 0) {
    std::fprintf(stderr,
                 "PASE: distributed_sort_mpi_int: global_n_hint=%lld != sum=%lld "
                 "(using sum)\n",
                 static_cast<long long>(global_n_hint),
                 static_cast<long long>(sum_n));
  }

  const long long N = sum_n;
  if (N <= 0) {
    local_chunk.clear();
    return;
  }

  const std::size_t gather_bytes =
      static_cast<std::size_t>(N) * sizeof(int);
  if (gather_bytes > distributed_sort_mpi_gather_cap_bytes()) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "PASE: distributed_sort_mpi_int: N=%lld ints (%.1f MiB) "
                   "exceeds gather cap; sorting locally per rank.\n",
                   static_cast<long long>(N),
                   static_cast<double>(gather_bytes) / (1024.0 * 1024.0));
    }
    sort_locally(local_chunk);
    return;
  }

  std::vector<int> counts(static_cast<std::size_t>(size));
  MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  std::vector<int> displs(static_cast<std::size_t>(size));
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[static_cast<std::size_t>(i)] =
        displs[static_cast<std::size_t>(i - 1)] +
        counts[static_cast<std::size_t>(i - 1)];
  }

  std::vector<int> all;
  if (rank == 0) {
    all.resize(static_cast<std::size_t>(N));
  }

  int* send_buf = local_n > 0 ? local_chunk.data() : nullptr;
  int* recv_buf =
      (rank == 0 && static_cast<std::size_t>(N) > 0) ? all.data() : nullptr;
  MPI_Gatherv(send_buf, local_n, MPI_INT, recv_buf, counts.data(),
              displs.data(), MPI_INT, 0, comm);

  if (rank == 0) {
    adaptive_sort(all.data(), static_cast<int>(N));
  }

  std::vector<int> out_counts(static_cast<std::size_t>(size));
  std::vector<int> out_displs(static_cast<std::size_t>(size));
  if (rank == 0) {
    long long acc = 0;
    for (int i = 0; i < size; ++i) {
      out_counts[static_cast<std::size_t>(i)] =
          static_cast<int>(N / size +
                          (i < static_cast<int>(N % size) ? 1 : 0));
      out_displs[static_cast<std::size_t>(i)] = static_cast<int>(acc);
      acc += out_counts[static_cast<std::size_t>(i)];
    }
  }
  MPI_Bcast(out_counts.data(), size, MPI_INT, 0, comm);
  MPI_Bcast(out_displs.data(), size, MPI_INT, 0, comm);

  const int my_out = out_counts[static_cast<std::size_t>(rank)];
  std::vector<int> scattered(
      static_cast<std::size_t>(std::max(0, my_out)));

  int* scatter_recv = my_out > 0 ? scattered.data() : nullptr;
  int* scatter_send =
      (rank == 0 && static_cast<std::size_t>(N) > 0) ? all.data() : nullptr;
  MPI_Scatterv(scatter_send, out_counts.data(), out_displs.data(), MPI_INT,
               scatter_recv, my_out, MPI_INT, 0, comm);

  local_chunk = std::move(scattered);
}

#endif

}  // namespace pase
