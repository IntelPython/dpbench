#include <cstdlib>
#include <CL/sycl.hpp>

#define SIZE3 4
#define NOUT 4
#define SEED 777

double* rambo(cl::sycl::queue*, size_t);
