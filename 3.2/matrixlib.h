#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NULL_CSR \
    (CSR_t) { NULL, NULL, NULL }

/* Structure used to contain a Compressed Sparse Row representation of a sparse matrix*/
typedef struct CSR {
    int *val_array; // Array of non-zero values in the matrix
    int *col_array; // Array of column indexes of non-zero values in the matrix
    int *start_idx; // Array of indexes to the start of each row(first non-zero value)
} CSR_t;

/*Builds the Compressed Sparse Row representation of a sparse matrix*/
CSR_t CSR_create(int **matrix, int row, int col, int non_zero);

/* Returns the product of multiplication between a matrix and a vector*/
int *mat_vec(int **matrix, int *vector, int row, int col);

/* Returns the product of multiplication between a matrix and a vector, using CSR representation*/
int *CSR_mat_vec(CSR_t rep, int *vec, int dimension);

/* Prints a given 2D matrix in the terminal */
void print_matrix(int **matrix, int row, int col);

/*Prints a CSR representation to terminal*/
void print_CSR(CSR_t csr, int rows);

/*Prints the elements of an integer array*/
void print_array(int *array, int len);

/*Parallel implementations using OpenMP*/

/*Builds the Compressed Sparse Row representation of a sparse matrix using parallel execution*/
int CSR_create_mpi(int *matrix_block, CSR_t *private_csr, int row, int row_block, int col, int process_count, MPI_Comm comm);

/* Returns the product of multiplication between a matrix and a vector using parallel execution*/
int *mat_vec_mpi(int *matrix_block, int *private_vector, int *private_result, int rows, int row_block, int col, int col_block, int process_count, MPI_Comm comm);

/* Returns the product of multiplication between a matrix and a vector using CSR representation, and parallel execution*/
int CSR_mat_vec_mpi(CSR_t *private_csr, int *private_vector, int *private_result, int rows, int row_block, int col, int col_block, int process_count, MPI_Comm comm);

/* Compares two integer arrays. Returns number of non-matching elements */
void compare_array(int *A1, int *A2, int dimension);

/* Compares two CSR representations, prints counts of non-matching elements */
void compare_CSR(CSR_t CSR1, CSR_t CSR2, int non_zero, int rows);

/* Deallocates the memory used by a CSR_t struct*/
int CSR_Destroy(CSR_t *csr);
