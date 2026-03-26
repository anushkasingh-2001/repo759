#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 2 || size != 2) {
    if (rank == 0) {
      std::cerr << "Usage: srun -n 2 ./task3 n\n";
    }
    MPI_Finalize();
    return 1;
  }

  const int n = std::stoi(argv[1]);
  std::vector<float> sendbuf(n), recvbuf(n, 0.0f);

  for (int i = 0; i < n; i++) {
    sendbuf[i] = static_cast<float>(rank + 1) + 0.001f * static_cast<float>(i);
  }

  const int peer = (rank == 0) ? 1 : 0;
  const int tag = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  double local_ms = 0.0;

  if (rank == 0) {
    double start = MPI_Wtime();
    MPI_Send(sendbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD);
    MPI_Recv(recvbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double end = MPI_Wtime();
    local_ms = (end - start) * 1000.0;
  } else if (rank == 1) {
    double start = MPI_Wtime();
    MPI_Recv(recvbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(sendbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD);
    double end = MPI_Wtime();
    local_ms = (end - start) * 1000.0;
  }

  if (rank == 1) {
    MPI_Send(&local_ms, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  } else {
    double other_ms = 0.0;
    MPI_Recv(&other_ms, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << (local_ms + other_ms) << "\n";
  }

  MPI_Finalize();
  return 0;
}