MAKEFLAGS += --no-print-directory
CFLAGS = -g -Wall 
LIBFLAGS = -lm

D = 1000  #ARRAY_DIMENSION
Z = 60   #ZERO_PERCENTAGE
R = 5    #REPETITIONS

ARGS = $(D) $(Z) $(R)

SAMPLES = 5 #Number of sample executions for test script


.SILENT:
test: compile-test
	chmod +x ./test_script.sh
	./test_script.sh $(SAMPLES)
	
compile-test: test.c
	gcc $(CFLAGS) test.c -o test

main: main.c matrixlib.c
	mpicc $(CFLAGS) -c main.c -fopenmp -o main.o
	mpicc $(CFLAGS) -c matrixlib.c -fopenmp -o matrixlib.o
	mpicc $(CFLAGS) main.o matrixlib.o -fopenmp -o main $(LIBFLAGS)

run: main
	mpiexec -n 4 ./main $(ARGS)

clean: main
	echo "Cleaning"
	rm main
