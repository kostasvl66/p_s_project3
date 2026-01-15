#include "matrixlib.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int process_count;

/*Returns a random number in the range 0 - <range_max>*/
int rand_from_range(int range_max) {
    int result = rand() / (RAND_MAX / range_max + 1);
    return result;
}

/*Return the amount of time elapsed between two timespec instances*/
double time_elapsed(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char *argv[]) {
    // Checking for correct number of arguments
    if (argc != 4) {
        fprintf(stderr, "Program must be called as -> ./main dimension zero_percentage reps\n");
        fprintf(stderr, "Can also be called as -> make run D=dimension Z=zero_percentage R=reps\n");
        fprintf(stderr, "Or as(using default values) -> make run\n");
        return 1;
    }

    // MPI initialization
    int my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Receiving program inputs
    int dimension = atoi(argv[1]);       // Matrix dimension
    int zero_percentage = atoi(argv[2]); // Percentage of matrix elements with a value of 0
    int reps = atoi(argv[3]);            // Number of times multiplication is repeated

    // Timespec initialization
    struct timespec serial_CSRrep_start, serial_CSRrep_finish;
    struct timespec serial_mult_start, serial_mult_finish;
    struct timespec serial_CSRmult_start, serial_CSRmult_finish;

    struct timespec parallel_CSRrep_start, parallel_CSRrep_finish;
    struct timespec parallel_mult_start, parallel_mult_finish;
    struct timespec parallel_CSRmult_start, parallel_CSRmult_finish;

    double serial_mult_elapsed;      // Time of serial multiplication
    double serial_CSR_elapsed;       // Time of serial CSR creation
    double serial_CSRmult_elapsed;   // Time of serial CSR multiplication
    double parallel_mult_elapsed;    // Time of paralle multiplication
    double parallel_CSR_elapsed;     // Time of parallel CSR creation
    double parallel_CSRmult_elapsed; // Time of parallel CSR multiplication

    // Seeding rand for consistent results during program execution
    srand(1);

    // Useful values
    int total_values = dimension * dimension;
    int zeroes = ceil(total_values * zero_percentage / 100.0);

    // Calculating number of non-zero values in the matrix
    int non_zero = total_values - zeroes;

    // Initializing pointers for later use
    int **mat = NULL;
    int *mat_mpi = NULL;
    int *vec = NULL;
    int *x = NULL;
    int *serial_res = NULL;
    int *serial_CSRres = NULL;
    CSR_t M_rep = NULL_CSR;
    int *parallel_res = NULL;
    int *parallel_CSRres = NULL;
    CSR_t parallel_M_rep = NULL_CSR;

    /* -------------------- Matrix/vector allocation -------------------- */

    if (my_rank == 0) {
        // Generate matrix for multiplication with values from 0-100
        mat = (int **)malloc(dimension * sizeof(int *));
        mat_mpi = (int *)malloc(dimension * dimension * sizeof(int));
        for (int x = 0; x < dimension; x++) {
            mat[x] = (int *)malloc(dimension * sizeof(int));
        }

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                // Setting all values of the matrix to random integers, excluding 0
                int val = rand_from_range(100);
                while (val == 0) {
                    val = rand_from_range(100);
                }
                mat[i][j] = val;
                mat_mpi[i * dimension + j] = val;
            }
        }

        // Setting random matrix values to 0
        for (int l = 0; l < zeroes; l++) {
            int row, col;

            row = rand_from_range(dimension);
            col = rand_from_range(dimension);

            while (mat[row][col] == 0) {
                // Make sure to pick a new value if the one checked is already 0
                row = rand_from_range(dimension);
                col = rand_from_range(dimension);
            }

            mat[row][col] = 0;
            mat_mpi[row * dimension + col] = 0;
        }

        // Create vector with random int values
        vec = (int *)malloc(dimension * sizeof(int));

        for (int k = 0; k < dimension; k++) {
            int val = rand_from_range(100);
            vec[k] = val;
        }
    }

    /* -------------------- Serial program execution -------------------- */

    if (my_rank == 0) {

        // Create CSR representation of sparse matrix
        timespec_get(&serial_CSRrep_start, TIME_UTC);

        M_rep = CSR_create(mat, dimension, dimension, non_zero);

        timespec_get(&serial_CSRrep_finish, TIME_UTC);

        // Storing elapsed time
        serial_CSR_elapsed = time_elapsed(serial_CSRrep_start, serial_CSRrep_finish);

        // Receiving product of matrix and vector using serial execution
        // The product of each repetition is set as the multiplication vector of the next one
        timespec_get(&serial_mult_start, TIME_UTC);
        serial_res = (int *)malloc(dimension * sizeof(int));
        x = vec;
        for (int repetition = 0; repetition < reps; repetition++) {
            serial_res = mat_vec(mat, x, dimension, dimension);
            x = serial_res;
        }

        timespec_get(&serial_mult_finish, TIME_UTC);

        // Storing elapsed time
        serial_mult_elapsed = time_elapsed(serial_mult_start, serial_mult_finish);

        // Receiving product of matrix and vector using serial execution
        // The product of each repetition is set as the multiplication vector of the next one
        timespec_get(&serial_CSRmult_start, TIME_UTC);
        serial_CSRres = (int *)malloc(dimension * sizeof(int));
        x = vec;
        for (int repetition = 0; repetition < reps; repetition++) {
            serial_CSRres = CSR_mat_vec(M_rep, vec, dimension);
            x = serial_CSRres;
        }

        timespec_get(&serial_CSRmult_finish, TIME_UTC);

        serial_CSRmult_elapsed = time_elapsed(serial_CSRmult_start, serial_CSRmult_finish);
    }

    /* -------------------- Parallel program execution -------------------- */

    // Base number of lines per process, not taking remainder into account
    int base_block = dimension / process_count;

    // Calculating remainder of rows and distributing them evenly across all processes
    int row_block_remainder = dimension % process_count;
    int row_block = base_block + (my_rank < row_block_remainder ? 1 : 0);

    // Calculating remainder of columns and distributing them evenly across all processes
    int col_block_remainder = dimension % process_count;
    int col_block = base_block + (my_rank < col_block_remainder ? 1 : 0);

    // Allocating memory for arrays used in multiplication
    int *mat_block = malloc(row_block * dimension * sizeof(int));

    // Allocating global struct to receive the final CSR representation
    CSR_t global_csr = NULL_CSR;

    if (my_rank == 0) {
        global_csr.col_array = malloc(non_zero * sizeof(int));
        global_csr.val_array = malloc(non_zero * sizeof(int));
        global_csr.start_idx = malloc((dimension + 1) * sizeof(int));
        global_csr.start_idx[dimension] = non_zero;
    }

    // Private CSR used by each process
    CSR_t private_csr = NULL_CSR;

    // Arrays to be used as arguments in Scatterv
    int *sendcounts = NULL;
    int *send_displacements = NULL;

    if (my_rank == 0) {
        sendcounts = malloc(process_count * sizeof(int));
        send_displacements = malloc(process_count * sizeof(int));

        int current_send_disp = 0;

        for (int i = 0; i < process_count; i++) {
            int row_block_per_process = base_block + (i < row_block_remainder ? 1 : 0);

            // Rows to be scattered from matrix
            sendcounts[i] = row_block_per_process * dimension;
            send_displacements[i] = current_send_disp;
            current_send_disp += sendcounts[i];
        }
    }

    // Scattering matrix(in contiguous form) from process 0 to the rest
    MPI_Scatterv(mat_mpi, sendcounts, send_displacements, MPI_INT, mat_block, row_block * dimension, MPI_INT, 0, MPI_COMM_WORLD);

    // Create CSR representation of sparse matrix using parallel execution
    if (my_rank == 0) {
        timespec_get(&parallel_CSRrep_start, TIME_UTC);
    }

    CSR_create_mpi(mat_block, &private_csr, dimension, row_block, dimension, process_count, MPI_COMM_WORLD);

    if (my_rank == 0) {
        timespec_get(&parallel_CSRrep_finish, TIME_UTC);

        // Storing elapsed time
        parallel_CSR_elapsed = time_elapsed(parallel_CSRrep_start, parallel_CSRrep_finish);
    }
    int *idx_recvcounts;
    int *idx_displacements;

    if (my_rank == 0) {
        idx_recvcounts = malloc(process_count * sizeof(int));
        idx_displacements = malloc(process_count * sizeof(int));

        int current_disp = 0;

        for (int i = 0; i < process_count; i++) {
            int row_block_per_process = base_block + (i < row_block_remainder ? 1 : 0);

            idx_recvcounts[i] = row_block_per_process;
            idx_displacements[i] = current_disp;
            current_disp += idx_recvcounts[i];
        }
    }

    // Gathering list of row-start indexes
    MPI_Gatherv(private_csr.start_idx, row_block, MPI_INT, global_csr.start_idx, idx_recvcounts, idx_displacements, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        print_array(global_csr.start_idx, dimension + 1);
    }

    // Receiving product of matrix and vector using parallel execution
    // The product of each repetition is set as the multiplication vector of the next one

    free(mat_block);
    mat_block = malloc(row_block * dimension * sizeof(int));
    int *priv_vec = malloc(col_block * sizeof(int));
    int *priv_res = malloc(row_block * sizeof(int));

    // Allocating result vector for each repetition
    if (my_rank == 0) {
        parallel_res = malloc(dimension * sizeof(int));
    }

    // Scatterv/Gatherv initial setup

    // Arrays to be used as arguments in Gatherv
    int *recvcounts = NULL;
    int *recv_displacements = NULL;

    if (my_rank == 0) {
        // Allocating arrays
        recvcounts = malloc(process_count * sizeof(int));
        recv_displacements = malloc(process_count * sizeof(int));

        int current_recv_disp = 0;

        // Per the number of processes, finding the number of rows each process must calculate
        // and setting each element of sendcounts to said value, making sure the displacements are also set.
        for (int i = 0; i < process_count; i++) {
            int row_block_per_process = base_block + (i < row_block_remainder ? 1 : 0);

            // Rows to be scattered from vector
            recvcounts[i] = row_block_per_process;
            recv_displacements[i] = current_recv_disp;
            current_recv_disp += recvcounts[i];
        }
    }

    // Scattering matrix(in contiguous form) from process 0 to the rest
    MPI_Scatterv(mat_mpi, sendcounts, send_displacements, MPI_INT, mat_block, row_block * dimension, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        timespec_get(&parallel_mult_start, TIME_UTC);
    }
    x = vec;

    for (int rep = 0; rep < reps; rep++) {
        // Scattering vector from process 0 to the rest
        MPI_Scatterv(x, recvcounts, recv_displacements, MPI_INT, priv_vec, col_block, MPI_INT, 0, MPI_COMM_WORLD);

        mat_vec_mpi(mat_block, priv_vec, priv_res, dimension, row_block, dimension, col_block, process_count, MPI_COMM_WORLD);

        // Gathering final vector of repetition back into process 0
        MPI_Gatherv(priv_res, row_block, MPI_INT, parallel_res, recvcounts, recv_displacements, MPI_INT, 0, MPI_COMM_WORLD);

        x = parallel_res;
    }

    if (my_rank == 0) {
        timespec_get(&parallel_mult_finish, TIME_UTC);

        // Storing elapsed time
        parallel_mult_elapsed = time_elapsed(parallel_mult_start, parallel_mult_finish);
    }

    free(priv_vec);
    free(priv_res);
    free(mat_block);

    // Receiving product of matrix and vector using serial execution
    // The product of each repetition is set as the multiplication vector of the next one
    if (my_rank == 0) {
        timespec_get(&parallel_CSRmult_start, TIME_UTC);

        parallel_CSRres = (int *)malloc(dimension * sizeof(int));
    }
    x = vec;
    for (int repetition = 0; repetition < reps; repetition++) {
        parallel_CSRres = CSR_mat_vec_mpi(parallel_M_rep, vec, dimension);
        x = parallel_CSRres;
    }

    if (my_rank == 0) {
        timespec_get(&parallel_CSRmult_finish, TIME_UTC);

        parallel_CSRmult_elapsed = time_elapsed(parallel_CSRmult_start, parallel_CSRmult_finish);
    }

    // HACK: Comparing CSR reps and result vectors
    if (my_rank == 0) {
        printf("Result vectors of serial and parallel multiplication: ");
        compare_array(serial_res, parallel_res, dimension);
        // print_array(serial_res, dimension);
        // print_array(parallel_res, dimension);
        printf("\n");

        compare_CSR(M_rep, parallel_M_rep, non_zero, dimension);

        printf("Result vectors of serial and parallel CSR multiplication: ");
        compare_array(serial_CSRres, parallel_CSRres, dimension);
        printf("\n");
    }

    if (my_rank == 0) {
        printf("File operation\n");
        // Writing data to external file
        FILE *fd;
        fd = fopen("test_data.txt", "a");
        if (fd == NULL) {
            perror("Error opening file");
        }

        int program_parameters[4] = {
            dimension,       // Matrix dimension
            zero_percentage, // Percentage of matrix elements with a value of 0
            reps,            // Number of times multiplication is repeated
        };
        double program_outputs[6] = {
            serial_mult_elapsed,     // Time of serial multiplication
            serial_CSR_elapsed,      // Time of serial CSR creation
            serial_CSRmult_elapsed,  // Time of serial CSR multiplication
            parallel_mult_elapsed,   // Time of paralle multiplication
            parallel_CSR_elapsed,    // Time of parallel CSR creation
            parallel_CSRmult_elapsed // Time of parallel CSR multiplication
        };

        // Writing program parameters to external file for testing purposes
        for (int parameter = 0; parameter < 4; parameter++) {
            fprintf(fd, "%d\n", program_parameters[parameter]);
        }

        // Writing program outputs to external file for testing purposes
        for (int output = 0; output < 6; output++) {
            fprintf(fd, "%lf\n", program_outputs[output]);
        }

        // Clearing memory
        fclose(fd);
    }

    // Memory allocated specifically by Process 0 should also be freed by it and no other
    if (my_rank == 0) {
        free(serial_res);
        serial_res = NULL;
        free(serial_CSRres);
        serial_CSRres = NULL;
        for (int row = 0; row < dimension; row++) {
            free(mat[row]);
        }
        free(mat);
        mat = NULL;
        free(vec);
        vec = NULL;
        free(parallel_res);
        parallel_res = NULL;
        free(parallel_CSRres);
        parallel_CSRres = NULL;
        CSR_destroy(&M_rep);
    }
    CSR_destroy(&parallel_M_rep);

    MPI_Finalize();

    return 0;
}
