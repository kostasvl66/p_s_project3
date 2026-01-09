#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>

// each polynomial coefficient a non-zero int within [-MAX_ABS_COEFFICIENT_VALUE, +MAX_ABS_COEFFICIENT_VALUE]
#define MAX_ABS_COEFFICIENT_VALUE 1000

// only for local use in pol_multiply_parallel();
// automates NULL checking and handling for malloc() or calloc() results
#define SAFE_CALL(call) \
if (!(call)) \
{ \
    success = 0; \
    goto cleanup; \
}

int N;

// checks command usage; returns 0 on success, -1 otherwise
int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <pol degree>\n", argv[0]);
        return -1;
    }
        
    char *end;
    long temp_n = strtol(argv[1], &end, 10);
    if (*end != '\0' || temp_n < 0 || temp_n > INT_MAX)
    {
        fprintf(stderr, "Invalid polynomial degree.\n");
        return -1;
    }
 
    N = (int)temp_n;
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
    int degree;    // the largest power
} Polynomial;

// returns a random non-zero coefficient array of length (degree + 1) or NULL if the allocation fails
// the caller receives ownership of the allocated memory
// srand() must be called before the function call
int *generate_random_coef(int degree)
{
    int *coef_arr = malloc((degree + 1) * sizeof(int));
    if (!coef_arr) return NULL;

    for (int i = 0; i <= degree; i++)
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
int pol_init(Polynomial **out_pol, int *coef_arr, int degree)
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
    for (int i = pol->degree; i >= 0; i--)
    {
        printf("(%d)x^%d", pol->coef_arr[i], i);
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
    
    for (int i = 0; i <= pol1->degree; i++)
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
    int degree1 = pol1->degree;
    int degree2 = pol2->degree;

    Polynomial *max_pol; // polynomial with the largest degree
    int min_degree; // smallest of the two degrees
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
    
    for (int i = 0; i <= min_degree; i++)
        result->coef_arr[i] = pol1->coef_arr[i] + pol2->coef_arr[i];

    for (int i = min_degree + 1; i <= result->degree; i++)
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

    for (int i = 0; i <= pol1->degree; i++)
    {
        prod_i.degree = pol2->degree + i;
        for (int j = 0; j <= prod_i.degree; j++)
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

// called by every rank; all communications happen inside this call
// non-zero ranks should pass NULL in pol1, pol2
// for rank 0 *result gets the product, everyone else gets NULL
// if *result != NULL, it is caller's responsibility to pol_destroy(result)
// returns 0 on success, -1 otherwise
int pol_multiply_parallel(
    Polynomial *pol1,
    Polynomial *pol2,
    Polynomial **result,
    int comm_rank,
    int comm_size
) {
    int success = 1;

    // initializing to NULL for safe cleanup
    Polynomial *res      = NULL;
    int *sendcounts      = NULL;
    int *displs          = NULL;
    int *packed_chunks   = NULL;
    int *local_coefs1    = NULL;
    Polynomial *prod_i   = NULL;
    int *prod_i_coef_arr = NULL;
    Polynomial *acc      = NULL;
    int *acc_coef_arr    = NULL;

    int deg1, deg2, final_deg;
    int degs[2]; // deg1 and deg2
    if (comm_rank == 0)
    {   
        degs[0] = pol1->degree;
        degs[1] = pol2->degree;
    }

    MPI_Bcast(degs, 2, MPI_INT, 0, MPI_COMM_WORLD);
    deg1 = degs[0];
    deg2 = degs[1];
    final_deg = deg1 + deg2;

    if (comm_rank == 0)
    {   
        // res is only relevant to rank 0
        SAFE_CALL(res = malloc(sizeof(Polynomial)));

        // new degree is the sum of the two
        res->degree = pol1->degree + pol2->degree;
        SAFE_CALL(res->coef_arr = calloc(res->degree + 1, sizeof(int))); // initialized at 0
    }
    else
    {
        // for non-zero ranks, pol2 will store the received coefficients of original pol2
        SAFE_CALL(pol2 = malloc(sizeof(Polynomial)));
        SAFE_CALL(pol2->coef_arr = malloc((deg2 + 1) * sizeof(int)));

        pol2->degree = deg2;
    }

    if (comm_rank == 0)
    {
        // preparing to scatter coefficients packed in chunks of common remainder mod comm_size
        SAFE_CALL(sendcounts = calloc(comm_size, sizeof(int))); // initialized to 0
        SAFE_CALL(displs = malloc(comm_size * sizeof(int)));

        for (int i = 0; i <= deg1; i++)
            sendcounts[i % comm_size]++;

        displs[0] = 0;
        for (int i = 0; i < comm_size; i++)
            displs[i] = sendcounts[i - 1] + displs[i - 1];

        SAFE_CALL(packed_chunks = malloc((deg1 + 1) * sizeof(int)));

        int *pos; // pos[rank] == index in packed_chunks to place next element that rank gets
        SAFE_CALL(pos = malloc(comm_size * sizeof(int)));
        memcpy(pos, displs, comm_size * sizeof(int));

        for (int i = 0; i <= deg1; i++)
        {
            packed_chunks[pos[i % comm_size]] = pol1->coef_arr[i];
            pos[i % comm_size]++;
        }

        free(pos);
    }

    // scattering receive counts
    int local_n1;
    MPI_Scatter(
        sendcounts,
        1, 
        MPI_INT,
        &local_n1,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // getting local coefficients
    SAFE_CALL(local_coefs1 = malloc(local_n1 * sizeof(int)));
    MPI_Scatterv(
        packed_chunks,
        sendcounts,
        displs,
        MPI_INT,
        local_coefs1,
        local_n1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // broadcasting all pol2 coefficients
    MPI_Bcast(pol2->coef_arr, deg2 + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // stores the product of i-th pol1 term and the whole pol2
    SAFE_CALL(prod_i = malloc(sizeof(Polynomial)));

    // stores the coef_arr that will be assigned to prod_i_per_thread[thread]
    SAFE_CALL(prod_i_coef_arr = malloc((final_deg + 1) * sizeof(int)));

    // assigning coefficient arrays
    prod_i->coef_arr = prod_i_coef_arr;

    // stores an accumulator polynomial for partial sums, which are combined in the end to form res
    SAFE_CALL(acc = malloc(sizeof(Polynomial)));

    // stores the coef_arr that will be assigned to acc
    SAFE_CALL(acc_coef_arr = calloc(final_deg + 1, sizeof(int))); // initialized to 0

    // assigning coefficient arrays and degree (max degree needed)
    acc->coef_arr = acc_coef_arr;
    acc->degree = deg1 + deg2;

    int local_idx = 0;
    for (int i = comm_rank; i <= deg1; i += comm_size)
    {
        prod_i->degree = pol2->degree + i;
        for (int j = 0; j <= prod_i->degree; j++)
        {
            if (j < i)
                prod_i->coef_arr[j] = 0;
            else
                prod_i->coef_arr[j] = local_coefs1[local_idx] * pol2->coef_arr[j - i];
        }

        pol_add(prod_i, acc, acc);
        local_idx++;
    }

    if (comm_rank == 0)
    {
        // first adding local acc of itself to res
        pol_add(acc, res, res);
        
        // receiving other local sums and adding them
        for (int rank = 1; rank < comm_size; rank++)
        {
            MPI_Recv(acc->coef_arr, acc->degree + 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pol_add(acc, res, res);
        }
    }
    else
    {
        // sending local accumulated sum
        MPI_Send(acc->coef_arr, acc->degree + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

cleanup:
    free(acc_coef_arr);
    free(acc);
    free(prod_i_coef_arr);
    free(prod_i);
    free(local_coefs1);
    free(displs);
    free(sendcounts);
    
    if (comm_rank != 0)
        pol_destroy(&pol2);

    if (!success)
    {
        if (comm_rank == 0)
            pol_destroy(&res);
        return -1;
    }

    *result = (comm_rank == 0) ? res : NULL;
    return 0;
}

int main(int argc, char *argv[])
{
    int comm_size, my_rank, was_rank_0 = 0;
    struct timespec start_init, end_init, start_serial, end_serial, start_parallel, end_parallel;
    Polynomial *pol1 = NULL, *pol2 = NULL, *prod1, *prod2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0)
    {
        was_rank_0 = 1;
        timespec_get(&start_init, TIME_UTC);

        if (parse_args(argc, argv) == -1)
            MPI_Abort(MPI_COMM_WORLD, 1);
        
        srand(time(NULL));

        int *coef_arr1 = generate_random_coef(N);
        if (!coef_arr1) MPI_Abort(MPI_COMM_WORLD, 1);
        if (pol_init(&pol1, coef_arr1, N) == -1) MPI_Abort(MPI_COMM_WORLD, 1);
        //pol_print(pol1);

        int *coef_arr2 = generate_random_coef(N);
        if (!coef_arr2) MPI_Abort(MPI_COMM_WORLD, 1);
        if (pol_init(&pol2, coef_arr2, N) == -1)
        {
            pol_destroy(&pol1);
            MPI_Abort(MPI_COMM_WORLD, 1);
        } 
        //pol_print(pol2);

        timespec_get(&end_init, TIME_UTC);
        printf("Initialization:     %.6f sec\n", elapsed(start_init, end_init));

        timespec_get(&start_parallel, TIME_UTC);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // every rank calls this
    if (pol_multiply_parallel(pol1, pol2, &prod2, my_rank, comm_size) == -1)
        MPI_Abort(MPI_COMM_WORLD, 1);

    if (my_rank == 0)
    {
        if (!prod2)
        {
            pol_destroy(&pol1);
            pol_destroy(&pol2);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        //pol_print(prod2);

        timespec_get(&end_parallel, TIME_UTC);
    }
    
    MPI_Finalize();

    // serial test after MPI_Finalize() to avoid any overhead
    if (was_rank_0)
    {
        timespec_get(&start_serial, TIME_UTC);

        prod1 = pol_multiply(pol1, pol2);
        if (!prod1)
        {
            pol_destroy(&pol1);
            pol_destroy(&pol2);
            pol_destroy(&prod2);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        //pol_print(prod1);

        timespec_get(&end_serial, TIME_UTC);
        printf("Serial algorithm:   %.6f sec\n", elapsed(start_serial, end_serial));
        printf("Parallel algorithm: %.6f sec\n", elapsed(start_parallel, end_parallel));

        if (pol_equals(prod1, prod2))
            printf("Consistent results\n");
        else
            printf("Inconsistent results\n");
        
        pol_destroy(&pol1);
        pol_destroy(&pol2);
        pol_destroy(&prod1);
        pol_destroy(&prod2);
    }

    return 0;
}