#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // For potential NAN/INF demonstration

// This is an experiment to show the impact of not waiting on MPI_Isend/MPI_Irecv

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Simple data: each process has a local value
    double local_value = rank * 1.0;  // e.g., 0.0, 1.0, 2.0, etc.
    double received_value = 1.0;      // Buffer for receiving

    // Define neighbor: simple point-to-point, process i sends to i+1, receives from i-1
    // For simplicity, make it a ring: rank sends to (rank+1)%size, receives from (rank-1+size)%size
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    // Asynchronous send and receive
    MPI_Request send_request, recv_request;
    MPI_Isend(&local_value, 1, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD, &send_request);
    MPI_Irecv(&received_value, 1, MPI_DOUBLE, recv_from, 0, MPI_COMM_WORLD, &recv_request);

    // Intentionally NOT using MPI_Wait or MPI_Waitall here to demonstrate the issue
    // Proceed directly to computation, which may use uninitialized or garbage received_value

    // Simple computation: compute average with received value, which might cause NAN/INF if received_value is garbage
    double local_density = 1.0;  // Assume some density
    double computed_result = local_value / received_value;  // Potential divide by zero or garbage

    // If received_value is uninitialized (e.g., 0.0 or garbage), this could be INF or NAN
    if (rank == 0) {
        printf("Computed result on rank %d: %f\n", rank, computed_result);
        // Gather results from others for "collate" simulation
        for (int i = 1; i < size; i++) {
            double other_result;
            MPI_Recv(&other_result, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Computed result on rank %d: %f\n", i, other_result);
        }
    } else {
        MPI_Send(&computed_result, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    // Now, to show what happens if we DID wait (for comparison, but commented out)
    // MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    // MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
    // int flag = 0;
    // MPI_Test(&recv_request, &flag, MPI_STATUS_IGNORE);
    // ... then recompute

    MPI_Finalize();
    return 0;
}