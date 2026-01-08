#include "matrixlib.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int threads;

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
    if (argc != 5) {
        fprintf(stderr, "Program must be called as -> ./main dimension zero_percentage reps threads\n");
        fprintf(stderr, "Can also be called as -> make run D=dimension Z=zero_percentage R=reps T=threads\n");
        fprintf(stderr, "Or as(using default values) -> make run\n");
        return 1;
    }
    // Receiving program inputs
    int dimension = atoi(argv[1]);       // Matrix dimension
    int zero_percentage = atoi(argv[2]); // Percentage of matrix elements with a value of 0
    int reps = atoi(argv[3]);            // Number of times multiplication is repeated
    threads = atoi(argv[4]);             // Number of threads used for parallel execution

    // Timespec initialization
    struct timespec serial_CSRrep_start, serial_CSRrep_finish;
    struct timespec serial_mult_start, serial_mult_finish;
    struct timespec serial_CSRmult_start, serial_CSRmult_finish;

    struct timespec parallel_CSRrep_start, parallel_CSRrep_finish;
    struct timespec parallel_mult_start, parallel_mult_finish;
    struct timespec parallel_CSRmult_start, parallel_CSRmult_finish;

    // Seeding rand for consistent results during program execution
    srand(1);

    // Number of processes to be run with MPI
    int comm_sz, my_rank;

    // MPI initialization
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Useful values
    int total_values = dimension * dimension;
    int zeroes = ceil(total_values * zero_percentage / 100.0);

    // Calculating number of non-zero values in the matrix
    int non_zero = total_values - zeroes;

    // Generate matrix for multiplication with values from 0-100
    int **mat = (int **)malloc(dimension * sizeof(int *));
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
    }

    // Create vector with random int values
    int *vec = (int *)malloc(dimension * sizeof(int));

    for (int k = 0; k < dimension; k++) {
        int val = rand_from_range(100);
        vec[k] = val;
    }

    // Create CSR representation of sparse matrix
    timespec_get(&serial_CSRrep_start, TIME_UTC);
    CSR_t M_rep = CSR_create(mat, dimension, dimension, non_zero);
    timespec_get(&serial_CSRrep_finish, TIME_UTC);

    // Storing elapsed time
    double serial_CSR_elapsed = time_elapsed(serial_CSRrep_start, serial_CSRrep_finish);

    // Receiving product of matrix and vector using serial execution
    // The product of each repetition is set as the multiplication vector of the next one

    timespec_get(&serial_mult_start, TIME_UTC);
    int *serial_res = (int *)malloc(dimension * sizeof(int));
    int *x = vec;
    for (int repetition = 0; repetition < reps; repetition++) {
        serial_res = mat_vec(mat, x, dimension, dimension);
        x = serial_res;
    }

    timespec_get(&serial_mult_finish, TIME_UTC);

    // Storing elapsed time
    double serial_mult_elapsed = time_elapsed(serial_mult_start, serial_mult_finish);

    // Receiving product of matrix and vector using serial execution
    // The product of each repetition is set as the multiplication vector of the next one

    timespec_get(&serial_CSRmult_start, TIME_UTC);
    int *serial_CSRres = (int *)malloc(dimension * sizeof(int));
    x = vec;
    for (int repetition = 0; repetition < reps; repetition++) {
        serial_CSRres = CSR_mat_vec(M_rep, vec, dimension);
        x = serial_CSRres;
    }

    timespec_get(&serial_CSRmult_finish, TIME_UTC);

    double serial_CSRmult_elapsed = time_elapsed(serial_CSRmult_start, serial_CSRmult_finish);

    /* Parallel program execution */

    // Create CSR representation of sparse matrix using parallel execution
    timespec_get(&parallel_CSRrep_start, TIME_UTC);
    CSR_t parallel_M_rep = CSR_create_omp(mat, dimension, dimension, non_zero);
    timespec_get(&parallel_CSRrep_finish, TIME_UTC);

    // Storing elapsed time
    double parallel_CSR_elapsed = time_elapsed(parallel_CSRrep_start, parallel_CSRrep_finish);

    // Receiving product of matrix and vector using parallel execution
    // The product of each repetition is set as the multiplication vector of the next one
    timespec_get(&parallel_mult_start, TIME_UTC);
    int *parallel_res = (int *)malloc(dimension * sizeof(int));
    x = vec;
    for (int repetition = 0; repetition < reps; repetition++) {
        parallel_res = mat_vec_omp(mat, x, dimension, dimension);
        x = parallel_res;
    }

    timespec_get(&parallel_mult_finish, TIME_UTC);

    // Storing elapsed time
    double parallel_mult_elapsed = time_elapsed(parallel_mult_start, parallel_mult_finish);

    // Receiving product of matrix and vector using serial execution
    // The product of each repetition is set as the multiplication vector of the next one

    timespec_get(&parallel_CSRmult_start, TIME_UTC);
    int *parallel_CSRres = (int *)malloc(dimension * sizeof(int));
    x = vec;
    for (int repetition = 0; repetition < reps; repetition++) {
        parallel_CSRres = CSR_mat_vec_omp(parallel_M_rep, vec, dimension);
        x = parallel_CSRres;
    }

    timespec_get(&parallel_CSRmult_finish, TIME_UTC);

    double parallel_CSRmult_elapsed = time_elapsed(parallel_CSRmult_start, parallel_CSRmult_finish);

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
        threads,         // Number of threads used for parallel execution
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
    for (int row = 0; row < dimension; row++) {
        free(mat[row]);
    }
    free(mat);
    mat = NULL;
    free(vec);
    vec = NULL;
    free(serial_res);
    serial_res = NULL;
    free(parallel_res);
    parallel_res = NULL;
    free(serial_CSRres);
    serial_CSRres = NULL;
    free(parallel_CSRres);
    parallel_CSRres = NULL;
    CSR_destroy(&M_rep);
    CSR_destroy(&parallel_M_rep);

    MPI_Finalize();

    return 0;
}
