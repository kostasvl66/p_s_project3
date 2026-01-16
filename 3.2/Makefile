MAKEFLAGS += --no-print-directory
CFLAGS = -g -Wall 
LIBFLAGS = -lm

D = 1000  #ARRAY_DIMENSION
Z = 60   #ZERO_PERCENTAGE
R = 5    #REPETITIONS
N = 4	 #NUMBER OF NODES USED

ARGS = $(D) $(Z) $(R)

SAMPLES = 5 #Number of sample executions for test script


.SILENT:
test: compile-test
	chmod +x ./test_script.sh
	./test_script.sh $(SAMPLES) $(N)
	
compile-test: test.c
	gcc $(CFLAGS) test.c -o test

compile: main.c matrixlib.c
	mpicc $(CFLAGS) -c main.c -o main.o
	mpicc $(CFLAGS) -c matrixlib.c -o matrixlib.o
	mpicc $(CFLAGS) main.o matrixlib.o -o main $(LIBFLAGS)

run: compile
	mpiexec -n $(N) ./main $(ARGS)

setup:
	chmod +x ../setup_files/askdate.sh
	../setup_files/askdate.sh

clean: main.o matrixlib.o test_data.txt main
	echo "Cleaning"
	rm main main.o matrixlib.o test_data.txt
