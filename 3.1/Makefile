CC = mpicc
CFLAGS = -Wall -O2
N = 3
P = 1

PROGRAM = polynomials

$(PROGRAM): polynomials.c
	@$(CC) $(CFLAGS) $< -o $@

run-loc: $(PROGRAM)
	mpiexec -n $(P) -map-by hwthread -bind-to hwthread ./$(PROGRAM) $(N)

run-lab: $(PROGRAM)
	mpiexec -f ../setup_files/machines -n $(P) ./$(PROGRAM) $(N)

setup:
	bash ../setup_files/askdate.sh

clean:
	@rm -f $(PROGRAM)