#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int send_value = rank;
    int recv_value = -1;

    MPI_Request reqs[2];
    // Non-blocking send a value to the other process
    MPI_Irecv(&recv_value, 1, MPI_INT, 1-rank, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(&send_value, 1, MPI_INT, 1-rank, 0, MPI_COMM_WORLD, &reqs[1]);

#ifdef USE_WAIT
    // Wait for both operations to complete
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
#endif

    // print the received value
    printf("Rank %d received %d\n", rank, recv_value);

    MPI_Finalize();
    return 0;
}
