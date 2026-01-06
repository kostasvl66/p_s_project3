CC = gcc
CFLAGS = -Wall -O2 -fopenmp
ARGS = 3 1

PROGRAM = polynomials

$(PROGRAM): polynomials.c
	@$(CC) $(CFLAGS) $< -o $@

run: $(PROGRAM)
	./$(PROGRAM) $(ARGS)

clean:
	@rm -f *.o $(PROGRAM)