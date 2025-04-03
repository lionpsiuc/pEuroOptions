# Compiler
CC = gcc
CCFLAGS = -std=c2x -Wall -Wextra
MPICC = mpicc
MPIFLAGS = -std=c2x -Wall -Wextra

# Flags
GSL = -lgsl -lgslcblas
LDFLAGS = -lm
OMP = -fopenmp

# Source and object files
SRCS = $(wildcard *.c)
EXES = $(SRCS:.c=)

# Default target
all: $(EXES)

# Serial
mc: mc.c
	$(CC) $(CCFLAGS) -o $@ $< $(GSL) $(LDFLAGS)

#OpenMP
ompmc: ompmc.c
	$(CC) $(CCFLAGS) $(OMP) -o $@ $< $(GSL) $(LDFLAGS)

# MPI
mpimc: mpimc.c
	$(MPICC) $(MPIFLAGS) -o $@ $< $(GSL) $(LDFLAGS)

# Clean target
.PHONY: clean
clean:
	$(RM) $(EXES)
