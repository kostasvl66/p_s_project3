CC = mpicc
CFLAGS = -Wall -O2
N = 3
P = 1

PROGRAM = polynomials

$(PROGRAM): polynomials.c
	@$(CC) $(CFLAGS) $< -o $@

# ensuring each rank is assigned to a different hardware thread
run-loc: $(PROGRAM)
	mpiexec -n $(P) -map-by hwthread -bind-to hwthread ./$(PROGRAM) $(N)

# ensuring each rank is assigned to a different core of each node;
# up to 4 processes per node, which is configured in machines file
run-lab: $(PROGRAM)
	mpiexec -f ../setup_files/machines -n $(P) -map-by node -bind-to core ./$(PROGRAM) $(N)

setup:
	bash ../setup_files/askdate.sh

clean:
	@rm -f $(PROGRAM)