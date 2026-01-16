#include "matrixlib.h"
#include <stdlib.h>

extern int process_count;

/*Builds the Compressed Sparse Row representation of a sparse matrix*/
CSR_t CSR_create(int **matrix, int row, int col, int non_zero) {
    CSR_t csr = {NULL, NULL, NULL};
    csr.val_array = (int *)malloc(non_zero * sizeof(int));
    csr.col_array = (int *)malloc(non_zero * sizeof(int));
    csr.start_idx = (int *)malloc((row + 1) * sizeof(int));
    csr.start_idx[row] = non_zero; // Last element of row_start list must contain the number of non-zero elements in the given matrix

    int list_idx = 0; // Index to use for accessing the above lists

    // Iterating through all elements in the matrix and checking for non-zero values
    for (int i = 0; i < row; i++) {
        csr.start_idx[i] = list_idx; // Storing the index of the column where a row's first non-zero element is
        int val;
        for (int j = 0; j < col; j++) {
            val = matrix[i][j];
            if (val != 0) {
                csr.val_array[list_idx] = val; // Storing non-zero value
                csr.col_array[list_idx] = j;   // Storing non-zero value's column index
                list_idx++;                    // Incrementing index
            }
        }
    }

    return csr;
}

/* Returns the product of multiplication between a matrix and a vector*/
int *mat_vec(int **matrix, int *vector, int row, int col) {
    int *res_vec = (int *)malloc(col * sizeof(int)); // Vector to store product
    for (int i = 0; i < row; i++) {
        res_vec[i] = 0; // Initializing values of result vector as 0
        for (int j = 0; j < col; j++) {
            res_vec[i] += matrix[i][j] * vector[j];
        }
    }

    return res_vec;
}

/* Returns the product of multiplication between a matrix and a vector, using CSR representation*/
int *CSR_mat_vec(CSR_t rep, int *vec, int dimension) {
    // Iterating through all non-zero values stored in the CSR representation
    int row_start, row_end;
    int *res_vec = malloc(dimension * sizeof(int));
    for (int i = 0; i < dimension; i++) {
        // Setting pointers to the start and the end of each line
        row_start = rep.start_idx[i];
        row_end = rep.start_idx[i + 1];
        res_vec[i] = 0; // Initializing elements of result vector as 0
        for (int j = row_start; j < row_end; j++) {
            int column = rep.col_array[j];
            res_vec[i] += rep.val_array[j] * vec[column];
        }
    }

    return res_vec;
}

/* Prints a given 2D matrix in the terminal */
void print_matrix(int **matrix, int row, int col) {
    for (int i = 0; i < row; i++) {
        printf("%d. \t", i);
        for (int j = 0; j < col; j++) {
            printf("%d \t", matrix[i][j]);
        }
        printf("\n");
    }
}

/*Prints a CSR representation to terminal*/
void print_CSR(CSR_t csr, int rows) {
    if (csr.start_idx == NULL || csr.val_array == NULL || csr.col_array == NULL) {
        printf("NULL CSR can't be printed.\n");
        return;
    }
    int nz_num = csr.start_idx[rows];
    printf("Values\t");
    for (int i = 0; i < nz_num; i++) {
        printf("%d \t", csr.val_array[i]);
    }
    printf("\n");

    printf("Columns\t");
    for (int i = 0; i < nz_num; i++) {
        printf("%d \t", csr.col_array[i]);
    }
    printf("\n");

    printf("Rows\t");
    for (int i = 0; i < rows; i++) {
        printf("%d \t", csr.start_idx[i]);
    }
    printf("\n");
}

/*Prints the elements of an integer array*/
void print_array(int *array, int len) {
    if (array == NULL) {
        printf("NULL array can't be printed.\n");
        return;
    }
    for (int i = 0; i < len; i++) {
        printf("%d\t", array[i]);
    }
}

/*Parallel execution functions*/

/*Builds the Compressed Sparse Row representation of a sparse matrix using parallel execution*/
int CSR_create_mpi(int *matrix_block, CSR_t *private_csr, int row, int row_block, int col, int process_count, MPI_Comm comm) {
    // Initializing first element of index list as 0
    private_csr->start_idx = (int *)malloc((row_block + 1) * sizeof(int));
    int *start_idx = private_csr->start_idx;

    start_idx[0] = 0;

    int total_nz_count = 0;

    for (int i = 0; i < row_block; i++) {
        int nz_per_line = 0; // Initializing a counter of non-zero elements in each row of the matrix
        for (int j = 0; j < col; j++) {
            if (matrix_block[i * col + j] != 0) {
                nz_per_line++;
            }
        }
        total_nz_count += nz_per_line;
        start_idx[i + 1] = nz_per_line; // The count of each line is stored in the index after its own
    }

    private_csr->val_array = (int *)malloc(total_nz_count * sizeof(int));
    private_csr->col_array = (int *)malloc(total_nz_count * sizeof(int));

    int *val_array = private_csr->val_array;
    int *col_array = private_csr->col_array;

    for (int x = 0; x < row_block; x++) {
        // The count of each line must have the count of the previous one added to it
        // This is so that each index contains an accurate pointer to the very first element of each line
        start_idx[x + 1] += start_idx[x];
    }

    for (int i = 0; i < row_block; i++) {
        // Starting each line from the first non-zero element, as calcualted earlier
        int current_idx = start_idx[i];
        for (int j = 0; j < col; j++) {
            int val = matrix_block[i * col + j];
            if (val != 0) {
                val_array[current_idx] = val; // Storing non-zero value
                col_array[current_idx] = j;   // Storing non-zero value's column index
                current_idx++;                // Incrementing index
            }
        }
    }

    return 0;
}

/* Returns the product of multiplication between a matrix and a vector using parallel execution*/
int *mat_vec_mpi(int *matrix_block, int *private_vector, int *private_result, int rows, int row_block, int col, int col_block, int process_count, MPI_Comm comm) {
    // Gathering vector for multiplication in all processes using MPI_Allgatherv.
    // MPI_Allgatherv is preferred over MPI_Allgather in cases where the number of
    // matrix rows is not divisible by the number of processes.
    // Scattering is also performed by MPI_Scatterv for the same reason.
    int *vector = (int *)malloc(rows * sizeof(int));

    int *recvcounts = malloc(process_count * sizeof(int));
    int *recv_displacements = malloc(process_count * sizeof(int));

    int base_block = rows / process_count;
    int remainder = rows % process_count;

    int current_recv_disp = 0;
    for (int i = 0; i < process_count; i++) {
        int rows_per_process = base_block + (i < remainder ? 1 : 0);

        recvcounts[i] = rows_per_process;
        recv_displacements[i] = current_recv_disp;
        current_recv_disp += recvcounts[i];
    }

    MPI_Allgatherv(private_vector, col_block, MPI_INT, vector, recvcounts, recv_displacements, MPI_INT, comm);

    // Multiplication is performed using the process-specific matrix block, as it was 
    // scattered by main, and the multiplication vector as it was acquired by MPI_Allgatherv.
    for (int private_i = 0; private_i < row_block; private_i++) {
        private_result[private_i] = 0;
        for (int j = 0; j < col; j++) {
            private_result[private_i] += matrix_block[private_i * col + j] * vector[j];
        }
    }
    return 0;
}

/* Returns the product of multiplication between a matrix and a vector using CSR representation, and parallel execution*/
int CSR_mat_vec_mpi(CSR_t *private_csr, int *private_vector, int *private_result, int rows, int row_block, int col, int col_block, int process_count, MPI_Comm comm) {
    // Gathering vector for multiplication in all processes using MPI_Allgatherv.
    int *vector = (int *)malloc(rows * sizeof(int));

    int *recvcounts = malloc(process_count * sizeof(int));
    int *recv_displacements = malloc(process_count * sizeof(int));

    int base_block = rows / process_count;
    int remainder = rows % process_count;

    int current_recv_disp = 0;
    for (int i = 0; i < process_count; i++) {
        int rows_per_process = base_block + (i < remainder ? 1 : 0);

        recvcounts[i] = rows_per_process;
        recv_displacements[i] = current_recv_disp;
        current_recv_disp += recvcounts[i];
    }

    MPI_Allgatherv(private_vector, col_block, MPI_INT, vector, recvcounts, recv_displacements, MPI_INT, comm);

    // Iterating through all non-zero values stored in the CSR representation
    int row_start, row_end;
    for (int i = 0; i < row_block; i++) {
        // Setting pointers to the start and the end of each line
        row_start = private_csr->start_idx[i];
        row_end = private_csr->start_idx[i + 1];
        private_result[i] = 0; // Initializing elements of result vector as 0
        for (int j = row_start; j < row_end; j++) {
            int column = private_csr->col_array[j];
            private_result[i] += private_csr->val_array[j] * vector[column];
        }
    }

    return 0;
}

/* Deallocates the memory used by a CSR_t struct*/
int CSR_Destroy(CSR_t *csr) {
    if (csr->col_array)
        free(csr->col_array);
    csr->col_array = NULL;
    if (csr->val_array)
        free(csr->val_array);
    csr->val_array = NULL;
    if (csr->start_idx)
        free(csr->start_idx);
    csr->start_idx = NULL;
    return 0;
}

/* Compares two integer arrays. Returns number of non-matching elements */
void compare_array(int *A1, int *A2, int dimension) {
    if (A1 == NULL) {
        printf("Array comparison failure. A1 is NULL\n");
        return;
    }
    if (A2 == NULL) {
        printf("Array comparison failure. A2 is NULL\n");
        return;
    }
    int count = 0;
    for (int i = 0; i < dimension; i++) {
        if (A1[i] != A2[i]) {
            count++;
        }
    }
    printf("%d mistakes in array comparison.\n", count);
}

/* Compares two CSR representations, prints counts of non-matching elements */
void compare_CSR(CSR_t CSR1, CSR_t CSR2, int non_zero, int rows) {
    if (CSR1.val_array == NULL) {
        printf("CSR comparison failure. CSR1 is NULL\n");
        return;
    }
    if (CSR2.val_array == NULL) {
        printf("CSR comparison failure. CSR2 is NULL\n");
        return;
    }

    int val_count = 0;
    int col_count = 0;
    int start_count = 0;

    for (int i = 0; i < non_zero; i++) {
        if (CSR1.val_array[i] != CSR2.val_array[i]) {
            val_count++;
        }
        if (CSR1.col_array[i] != CSR2.col_array[i]) {
            col_count++;
        }
    }
    for (int j = 0; j < rows; j++) {
        if (CSR1.start_idx[j] != CSR2.start_idx[j]) {
            start_count++;
        }
    }
    printf("There are %d mistakes in val_array.\n", val_count);
    printf("There are %d mistakes in col_array.\n", col_count);
    printf("There are %d mistakes in start_idx.\n", start_count);
}
