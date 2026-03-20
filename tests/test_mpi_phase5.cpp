#include <distributed_sort.h>
#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace {

TEST(MpiPhase5, GlobalIntGatherSortScatter) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "Run with at least 2 MPI ranks (e.g. mpirun -n 2)";
  }

  const int orig_n = rank + 1;
  std::vector<int> local(static_cast<std::size_t>(orig_n));
  for (int i = 0; i < orig_n; ++i) {
    local[static_cast<std::size_t>(i)] =
        (rank * 1000 + i * 99 + size * 3) % 10007;
  }
  const std::vector<int> orig = local;

  pase::distributed_sort_mpi_int(MPI_COMM_WORLD, local, /*global_n_hint=*/-1);

  long long N = orig_n;
  MPI_Allreduce(MPI_IN_PLACE, &N, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  std::vector<int> counts_in(static_cast<std::size_t>(size));
  MPI_Allgather(&orig_n, 1, MPI_INT, counts_in.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  std::vector<int> displs_in(static_cast<std::size_t>(size));
  displs_in[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs_in[static_cast<std::size_t>(i)] =
        displs_in[static_cast<std::size_t>(i - 1)] +
        counts_in[static_cast<std::size_t>(i - 1)];
  }

  std::vector<int> flat_orig(static_cast<std::size_t>(N));
  MPI_Allgatherv(orig.data(), orig_n, MPI_INT, flat_orig.data(),
                 counts_in.data(), displs_in.data(), MPI_INT, MPI_COMM_WORLD);
  std::vector<int> ref = flat_orig;
  std::sort(ref.begin(), ref.end());

  const int out_n = static_cast<int>(local.size());
  std::vector<int> counts_out(static_cast<std::size_t>(size));
  MPI_Allgather(&out_n, 1, MPI_INT, counts_out.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  std::vector<int> displs_out(static_cast<std::size_t>(size));
  displs_out[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs_out[static_cast<std::size_t>(i)] =
        displs_out[static_cast<std::size_t>(i - 1)] +
        counts_out[static_cast<std::size_t>(i - 1)];
  }

  std::vector<int> got(static_cast<std::size_t>(N));
  MPI_Allgatherv(local.data(), out_n, MPI_INT, got.data(), counts_out.data(),
                 displs_out.data(), MPI_INT, MPI_COMM_WORLD);

  EXPECT_EQ(got, ref);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  MPI_Finalize();
  return rc;
}
