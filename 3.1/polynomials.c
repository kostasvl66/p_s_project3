#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// each polynomial coefficient a non-zero int within [-MAX_ABS_COEFFICIENT_VALUE, +MAX_ABS_COEFFICIENT_VALUE]
#define MAX_ABS_COEFFICIENT_VALUE 1000

long N;
int THREAD_COUNT;

// checks command usage; returns 0 on success, -1 otherwise
int parse_args(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <pol degree> <thread count>\n", argv[0]);
        return -1;
    }
        
    char *end;
    N = strtol(argv[1], &end, 10);
    if (*end != '\0' || N < 0)
    {
        fprintf(stderr, "Invalid polynomial degree.\n");
        return -1;
    }
        
    long temp_thread_count = strtol(argv[2], &end, 10);
    if (*end != '\0' || temp_thread_count < 1 || temp_thread_count > INT_MAX)
    {
        fprintf(stderr, "Invalid thread count.\n");
        return -1;
    }
    THREAD_COUNT = (int)temp_thread_count;

    return 0;
}

// calculates time difference between start and end with nsec precision
double elapsed(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

typedef struct
{
    int *coef_arr; // array of coefficients, each index maps to the same power
    long degree; // the largest power
} Polynomial;

// returns a random non-zero coefficient array of length (degree + 1) or NULL if the allocation fails
// the caller receives ownership of the allocated memory
// srand() must be called before the function call
int *generate_random_coef(long degree)
{
    int *coef_arr = malloc((degree + 1) * sizeof(int));
    if (!coef_arr) return NULL;

    for (long i = 0; i <= degree; i++)
    {
        // making sure each coefficient is non-zero
        int abs_val = 1 + rand() % MAX_ABS_COEFFICIENT_VALUE; // 1 <= abs_val <= MAX_ABS_COEFFICIENT_VALUE
        int sign = (rand() % 2) ? 1 : -1; // random sign
        coef_arr[i] = sign * abs_val; // 1 <= abs(coef_arr[i]) <= MAX_ABS_COEFFICIENT_VALUE
    }

    return coef_arr;
}

// initializes a polynomial; returns 0 on success, -1 otherwise
// coef_arr must be heap allocated and is now owned by pol_init; that array is to be freed by pol_destroy (or pol_init)
int pol_init(Polynomial **out_pol, int *coef_arr, long degree)
{
    *out_pol = malloc(sizeof(Polynomial));
    if (!(*out_pol))
    {
        free(coef_arr); // if malloc fails coef_arr must be freed as it is now owned by pol_init
        return -1;
    } 

    (*out_pol)->coef_arr = coef_arr; // assuming the coef_arr passed remains intact
    (*out_pol)->degree = degree;
    return 0;
}

// deallocates the polynomial's data
// safe to call with *pol == NULL (does nothing)
void pol_destroy(Polynomial **pol)
{
    if (*pol)
    {
        free((*pol)->coef_arr);
        free(*pol);
        *pol = NULL;
    }
}

// prints a polynomial for test purposes
void pol_print(Polynomial *pol)
{
    for (long i = pol->degree; i >= 0; i--)
    {
        printf("(%d)x^%ld", pol->coef_arr[i], i);
        if (i > 0)
            printf("+");
        else
            printf("\n");
    }
}

// returns 1 if pol1 and pol2 have the same degree and coefficients; 0 otherwise
int pol_equals(Polynomial *pol1, Polynomial *pol2)
{
    if (pol1->degree != pol2->degree) return 0;
    
    for (long i = 0; i <= pol1->degree; i++)
        if (pol1->coef_arr[i] != pol2->coef_arr[i])
            return 0;

    return 1;
}

// result receives the sum of two polynomials; returns 0 if successful, -1 otherwise
// result is expected to have allocated large enough coef_arr, and only the terms up to the degree are updated
// this function works properly even if pol1 or pol2 points to the same memory as result
void pol_add(Polynomial *pol1, Polynomial *pol2, Polynomial *result)
{
    // saving degrees in case pol1 or pol2 points to the same memory as result
    long degree1 = pol1->degree;
    long degree2 = pol2->degree;

    Polynomial *max_pol; // polynomial with the largest degree
    long min_degree; // smallest of the two degrees
    if (degree1 > degree2)
    {
        result->degree = degree1; // new degree is the largest of the two
        max_pol = pol1;
        min_degree = degree2;
    }
    else
    {
        result->degree = degree2;
        max_pol = pol2;
        min_degree = degree1;
    }
    
    for (long i = 0; i <= min_degree; i++)
        result->coef_arr[i] = pol1->coef_arr[i] + pol2->coef_arr[i];

    for (long i = min_degree + 1; i <= result->degree; i++)
        result->coef_arr[i] = max_pol->coef_arr[i];
}

// returns the product of two polynomials; its memory is to be freed using pol_destroy; returns NULL if unsuccessful
Polynomial *pol_multiply(Polynomial *pol1, Polynomial *pol2)
{
    Polynomial *res = malloc(sizeof(Polynomial));
    if (!res) return NULL;

    // new degree is the sum of the two
    res->degree = pol1->degree + pol2->degree;
    res->coef_arr = calloc(res->degree + 1, sizeof(int)); // initialized to 0
    if (!(res->coef_arr))
    {
        free(res);
        return NULL;
    }

    Polynomial prod_i; // the product of i-th pol1 term and the whole pol2
    prod_i.degree = 0; // for now
    prod_i.coef_arr = malloc((res->degree + 1) * sizeof(int)); // pre-allocating the final needed size to avoid allocations in the loop
    if (!(prod_i.coef_arr))
    {
        free(res->coef_arr);
        free(res);
        return NULL;
    }

    for (long i = 0; i <= pol1->degree; i++)
    {
        prod_i.degree = pol2->degree + i;
        for (long j = 0; j <= prod_i.degree; j++)
        {
            if (j < i)
                prod_i.coef_arr[j] = 0;
            else
                prod_i.coef_arr[j] = pol1->coef_arr[i] * pol2->coef_arr[j - i];
        }

        pol_add(&prod_i, res, res); // adding prod_i to res
    }

    free(prod_i.coef_arr);
    return res;
}

// allocates 2d array (coef_arr for each thread) used like arr[thread][coef_index]
// all coef_arrays have the same length (max_degree + 1)
// returns NULL if failed
int **allocate_coef_arr_per_thread(int thread_count, long max_degree)
{
    int **arr = malloc(thread_count * sizeof(int *));
    if (!arr) return NULL;

    arr[0] = malloc(thread_count * (max_degree + 1) * sizeof(int));
    if (!arr[0]) 
    {
        free(arr);
        return NULL;
    }

    for (int i = 1; i < thread_count; i++)
        arr[i] = arr[0] + i * (max_degree + 1);

    return arr;
}

// allocates 2d array (coef_arr for each thread) used like arr[thread][coef_index]
// all coef_arrays have the same length (max_degree + 1) and are zero-initialized
// returns NULL if failed
int **zero_allocate_coef_arr_per_thread(int thread_count, long max_degree)
{
    int **arr = malloc(thread_count * sizeof(int *));
    if (!arr) return NULL;

    arr[0] = calloc(thread_count * (max_degree + 1), sizeof(int));
    if (!arr[0]) 
    {
        free(arr);
        return NULL;
    }

    for (int i = 1; i < thread_count; i++)
        arr[i] = arr[0] + i * (max_degree + 1);

    return arr;
}

// frees 2d array allocated either by allocate_coef_arr_per_thread or zero_allocate_coef_arr_per_thread
// safe to call with NULL (does nothing)
void free_coef_arr_per_thread(int **coef_per_thread)
{
    if (coef_per_thread)
    {
        free(coef_per_thread[0]);
        free(coef_per_thread);
    }
}

Polynomial *pol_multiply_threaded(Polynomial *pol1, Polynomial *pol2, int thread_count)
{
    #ifndef _OPENMP
    fprintf(stderr, "OpenMP is not supported.\n");
    return NULL;
    #else

    int success = 1;

    // initializing to NULL for safe cleanup
    Polynomial *res                  = NULL;
    Polynomial *prod_i_per_thread    = NULL;
    int **prod_i_coef_arr_per_thread = NULL;
    Polynomial *acc_per_thread       = NULL;
    int **acc_coef_arr_per_thread    = NULL;

    res = malloc(sizeof(Polynomial));
    if (!res)
    {
        success = 0;
        goto cleanup;
    }

    // new degree is the sum of the two
    res->degree = pol1->degree + pol2->degree;
    res->coef_arr = calloc(res->degree + 1, sizeof(int)); // initialized at 0
    if (!(res->coef_arr))
    {
        success = 0;
        goto cleanup;
    }

    // stores for each thread, the product of i-th pol1 term and the whole pol2
    prod_i_per_thread = malloc(thread_count * sizeof(Polynomial));
    if (!prod_i_per_thread)
    {
        success = 0;
        goto cleanup;
    }

    // stores for each thread, the coef_arr that will be assigned to prod_i_per_thread[thread]
    prod_i_coef_arr_per_thread = allocate_coef_arr_per_thread(thread_count, res->degree);
    if (!prod_i_coef_arr_per_thread)
    {
        success = 0;
        goto cleanup;
    }

    for (int thread = 0; thread < thread_count; thread++) // assigns coefficient arrays
        prod_i_per_thread[thread].coef_arr = prod_i_coef_arr_per_thread[thread];

    // stores for each thread, an accumulator polynomial for partial sums, which are combined in the end to form res
    acc_per_thread = malloc(thread_count * sizeof(Polynomial));
    if (!acc_per_thread)
    {
        success = 0;
        goto cleanup;
    }

    // stores for each thread, the coef_arr that will be assigned to acc_per_thread[thread]
    acc_coef_arr_per_thread = zero_allocate_coef_arr_per_thread(thread_count, res->degree);
    if (!acc_coef_arr_per_thread)
    {
        success = 0;
        goto cleanup;
    }

    for (int thread = 0; thread < thread_count; thread++) // assigns coefficient arrays
        acc_per_thread[thread].coef_arr = acc_coef_arr_per_thread[thread];

    #pragma omp parallel num_threads(thread_count) \
        default(none) shared(pol1, pol2, res, prod_i_per_thread, acc_per_thread)
    {
        int rank = omp_get_thread_num();
        Polynomial *prod_i = &prod_i_per_thread[rank];
        Polynomial *acc = &acc_per_thread[rank];

        #pragma omp for schedule(static, 1) nowait
        for (long i = 0; i <= pol1->degree; i++)
        {
            prod_i->degree = pol2->degree + i;
            for (long j = 0; j <= prod_i->degree; j++)
            {
                if (j < i)
                    prod_i->coef_arr[j] = 0;
                else
                    prod_i->coef_arr[j] = pol1->coef_arr[i] * pol2->coef_arr[j - i];
            }

            pol_add(prod_i, acc, acc);
        }

        #pragma omp critical
        pol_add(acc, res, res); // adding acc to res
    }
    
cleanup:
    free_coef_arr_per_thread(acc_coef_arr_per_thread);
    free(acc_per_thread);
    free_coef_arr_per_thread(prod_i_coef_arr_per_thread);
    free(prod_i_per_thread);
    if (!success)
    {
        pol_destroy(&res);
        return NULL;
    }

    return res;
    #endif
}

int main(int argc, char *argv[])
{   
    struct timespec start_init, end_init, start_serial, end_serial, start_parallel, end_parallel;
    timespec_get(&start_init, TIME_UTC);

    if (parse_args(argc, argv) == -1) return 1;
    
    srand(time(NULL));

    int *coef_arr1 = generate_random_coef(N);
    if (!coef_arr1) return 1;
    Polynomial *pol1;
    if (pol_init(&pol1, coef_arr1, N) == -1) return 1;
    //pol_print(pol1);

    int *coef_arr2 = generate_random_coef(N);
    if (!coef_arr2) return 1;
    Polynomial *pol2;
    if (pol_init(&pol2, coef_arr2, N) == -1)
    {
        pol_destroy(&pol1);
        return 1;
    } 
    //pol_print(pol2);

    timespec_get(&end_init, TIME_UTC);
    printf("Initialization:     %.6f sec\n", elapsed(start_init, end_init));

    timespec_get(&start_serial, TIME_UTC);

    Polynomial *prod1 = pol_multiply(pol1, pol2);
    if (!prod1)
    {
        pol_destroy(&pol1);
        pol_destroy(&pol2);
        return 1;
    }
    //pol_print(prod1);

    timespec_get(&end_serial, TIME_UTC);
    printf("Serial algorithm:   %.6f sec\n", elapsed(start_serial, end_serial));

    timespec_get(&start_parallel, TIME_UTC);

    Polynomial *prod2 = pol_multiply_threaded(pol1, pol2, THREAD_COUNT);
    if (!prod2)
    {
        pol_destroy(&pol1);
        pol_destroy(&pol2);
        pol_destroy(&prod1);
        return 1;
    }
    //pol_print(prod2);

    timespec_get(&end_parallel, TIME_UTC);
    printf("Parallel algorithm: %.6f sec\n", elapsed(start_parallel, end_parallel));

    if (pol_equals(prod1, prod2))
        printf("Consistent results\n");
    else
        printf("Inconsistent results\n");
    
    pol_destroy(&pol1);
    pol_destroy(&pol2);
    pol_destroy(&prod1);
    pol_destroy(&prod2);
    
    return 0;
}