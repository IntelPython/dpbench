# Standard flags
CFLAGS := -Wall -O3 -g

# Linker flags
LDFLAGS = -lm
CC = dpcpp

# link objects(binaries) together
pathfinder: main.o 
	$(CC) main.o $(LDFLAGS) -o pathfinder

# compile main function file into object (binary)
main.o: main.cpp  kernel.sycl
	$(CC) $(CFLAGS) main.cpp -c -o main.o

clean:
	@rm -f pathfinder *.o *.out *~ *\#
