MAKEFLAGS += --no-print-directory
CFLAGS = -g -Wall 
LIBFLAGS = -lm

D = 1000  #ARRAY_DIMENSION
Z = 60   #ZERO_PERCENTAGE
R = 5    #REPETITIONS
T = 4    #THREAD_NUM

SAMPLES = 5 #Number of sample executions for test script


.SILENT:
test: compile-test
	chmod +x ./test_script.sh
	./test_script.sh $(SAMPLES)
	
compile-test: test.c
	gcc $(CFLAGS) test.c -o test

main: main.c matrixlib.c
	gcc $(CFLAGS) -c main.c -fopenmp -o main.o
	gcc $(CFLAGS) -c matrixlib.c -fopenmp -o matrixlib.o
	gcc $(CFLAGS) main.o matrixlib.o -fopenmp -o main $(LIBFLAGS)

run: main
	./main $(D) $(Z) $(R) $ $(T)

clean: main
	echo "Cleaning"
	rm main
