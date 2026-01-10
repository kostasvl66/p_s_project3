#include "matrixlib.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NULL_CSR {NULL, NULL, NULL}

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

    // MPI process count and rank index initialization
    int comm_sz, my_rank;

    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    printf("Processes:%d\n", comm_sz);

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

    // Create CSR representation of sparse matrix using parallel execution
    if (my_rank == 0) {
        timespec_get(&parallel_CSRrep_start, TIME_UTC);
    }

    parallel_M_rep = CSR_create_mpi(mat_mpi, dimension, dimension, non_zero);

    if (my_rank == 0) {
        timespec_get(&parallel_CSRrep_finish, TIME_UTC);

        // Storing elapsed time
        parallel_CSR_elapsed = time_elapsed(parallel_CSRrep_start, parallel_CSRrep_finish);
    }

    // Receiving product of matrix and vector using parallel execution
    // The product of each repetition is set as the multiplication vector of the next one
    if (my_rank == 0) {
        timespec_get(&parallel_mult_start, TIME_UTC);

        parallel_res = (int *)malloc(dimension * sizeof(int));
    }
    x = vec;
    for (int repetition = 0; repetition < reps; repetition++) {
        parallel_res = mat_vec_mpi(mat_mpi, x, dimension, dimension);
        x = parallel_res;
    }

    if (my_rank == 0) {
        timespec_get(&parallel_mult_finish, TIME_UTC);

        // Storing elapsed time
        parallel_mult_elapsed = time_elapsed(parallel_mult_start, parallel_mult_finish);
    }

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

        compare_CSR(M_rep, parallel_M_rep, non_zero, dimension);

        compare_array(serial_CSRres, parallel_CSRres, dimension);
    }

    if (my_rank == 0) {

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
