#include <stdio.h>
#include <stdlib.h>

enum parameter_names {
    dimension,       // Matrix dimension
    zero_percentage, // Percentage of matrix elements with a value of 0
    reps,            // Number of times multiplication is repeated
    nodes            // Number of nodes used in execution
};

enum output_times {
    serial_mult_avg,     // Average time of serial multiplication
    serial_CSR_avg,      // Average time of serial CSR creation
    serial_CSRmult_avg,  // Average time of serial CSR multiplication
    parallel_mult_avg,   // Average time of paralle multiplication
    parallel_CSR_avg,    // Average time of parallel CSR creation
    parallel_CSRmult_avg // Average time of parallel CSR multiplication
};

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Program must be called as -> ./test <number of sample executions>");
    }

    int samples = atoi(argv[1]);

    FILE *fd;
    fd = fopen("test_data.txt", "r");
    if (fd == NULL) {
        perror("Error opening file");
    }

    int parameters[3];
    double averages[6] = {0}, temp;

    // Reading program outputs samples times and storing their averages
    for (int sample = 0; sample < samples; sample++) {
        for (int param = 0; param < 3; param++) {
            fscanf(fd, "%d", &parameters[param]);
        }
        for (int output = 0; output < 6; output++) {
            fscanf(fd, "%lf", &temp);
            averages[output] += temp;
        }

        averages[sample] = averages[sample] / (double)samples;
    }

    printf("With program parameters:\n");
    printf("Array dimension: %d ", parameters[dimension]);
    printf("Zero_percentage: %d ", parameters[zero_percentage]);
    printf("Repetitions: %d ", parameters[reps]);

    printf("Average time calculations for %d sample executions are:\n", samples);
    printf("Serial matrix-vector multiplication: %lf\n", averages[serial_mult_avg]);
    printf("Serial creation of CSR representation: %lf\n", averages[serial_CSR_avg]);
    printf("Serial CSR-vector multiplication: %lf\n", averages[serial_CSRmult_avg]);
    printf("Parallel matrix-vector multiplication: %lf\n", averages[parallel_mult_avg]);
    printf("Parallel creation of CSR representation: %lf\n", averages[parallel_CSR_avg]);
    printf("Parallel CSR-vector multiplication: %lf\n", averages[parallel_CSRmult_avg]);

    return 0;
}
