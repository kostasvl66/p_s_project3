#include "matrixlib.h"
#ifdef _OPENMP
#include <omp.h>
#endif

extern int threads;

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
    for (int i = 0; i < len; i++) {
        printf("%d\t", array[i]);
    }
}

/*Parallel execution functions*/

/*Builds the Compressed Sparse Row representation of a sparse matrix using parallel execution*/
CSR_t CSR_create_omp(int **matrix, int row, int col, int non_zero) {
    CSR_t csr = {NULL, NULL, NULL};
    csr.val_array = (int *)malloc(non_zero * sizeof(int));
    csr.col_array = (int *)malloc(non_zero * sizeof(int));
    csr.start_idx = (int *)malloc((row + 1) * sizeof(int));

    // Initializing first element of index list as 0
    csr.start_idx[0] = 0;

#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < row; i++) {
        int nz_count = 0; // Initializing a counter of non-zero elements in each row of the matrix
        for (int j = 0; j < col; j++) {
            if (matrix[i][j] != 0) {
                nz_count++;
            }
        }
        csr.start_idx[i + 1] = nz_count; // The count of each line is stored in the index after its own
    }

    // Prallelising this loop would create unnecessary overhead
    for (int x = 0; x < row; x++) {
        // The count of each line must have the count of the previous one added to it
        // This is so that each index contains an accurate pointer to the very first element of each line
        csr.start_idx[x + 1] += csr.start_idx[x];
    }

#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < row; i++) {
        // Starting each line from the first non-zero element, as calcualted earlier
        int current_idx = csr.start_idx[i];
        for (int j = 0; j < col; j++) {
            int val = matrix[i][j];
            if (val != 0) {
                csr.val_array[current_idx] = val; // Storing non-zero value
                csr.col_array[current_idx] = j;   // Storing non-zero value's column index
                current_idx++;                    // Incrementing index
            }
        }
    }

    return csr;
}

/* Returns the product of multiplication between a matrix and a vector using parallel execution*/
int *mat_vec_omp(int **matrix, int *vector, int row, int col) {
    int *res_vec = (int *)malloc(col * sizeof(int)); // Vector to store product
    int i, j;
#pragma omp parallel for num_threads(threads) default(none) private(i, j) shared(res_vec, matrix, vector, row, col)
    for (i = 0; i < row; i++) {
        res_vec[i] = 0; // Initializing values of result vector as 0
        for (j = 0; j < col; j++) {
            res_vec[i] += matrix[i][j] * vector[j];
        }
    }
    return res_vec;
}

/* Returns the product of multiplication between a matrix and a vector using CSR representation, and parallel execution*/
int *CSR_mat_vec_omp(CSR_t rep, int *vec, int dimension) {
    // Iterating through all non-zero values stored in the CSR representation
    int i, j, row_start, row_end;
    int *res_vec = malloc(dimension * sizeof(int));

#pragma omp parallel for num_threads(threads) default(none) private(i, j, row_start, row_end) shared(res_vec, rep, vec, dimension)
    for (i = 0; i < dimension; i++) {
        // Setting pointers to the start and the end of each line
        row_start = rep.start_idx[i];
        row_end = rep.start_idx[i + 1];
        res_vec[i] = 0; // Initializing elements of result vector as 0
        for (j = row_start; j < row_end; j++) {
            int column = rep.col_array[j];
            res_vec[i] += rep.val_array[j] * vec[column];
        }
    }

    return res_vec;
}

/* Deallocates the memory used by a CSR_t struct*/
int CSR_destroy(CSR_t *csr) {
    free(csr->col_array);
    csr->col_array = NULL;
    free(csr->val_array);
    csr->val_array = NULL;
    free(csr->start_idx);
    csr->start_idx = NULL;
    return 0;
}

/* Compares two integer arrays. Returns number of non-matching elements */
int compare_array(int *A1, int *A2, int dimension) {
    int count = 0;
    for (int i = 0; i < dimension; i++) {
        if (A1[i] != A2[i]) {
            count++;
        }
    }
    if (count == 0) {
        return 0;
    } else {
        return count;
    }
}

/* Compares two CSR representations, prints counts of non-matching elements */
void compare_CSR(CSR_t CSR1, CSR_t CSR2, int non_zero, int rows) {
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
