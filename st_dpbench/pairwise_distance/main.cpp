/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "constants_header.h"
#include "rdtsc.h"
#include <CL/sycl.hpp>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cl::sycl;
using namespace std;

int main(int argc, char *argv[])
{
    size_t nopt = 44032, ndims = 3;
    tfloat *x1, *x2;
    tfloat *distance_op;

    clock_t t1 = 0, t2 = 0;

    queue q;

    /* Allocate arrays, generate input data */
    InitData(q, nopt, ndims, &x1, &x2, &distance_op);

    /* Warm up cycle */
    pairwise_distance(q, nopt, x1, x2, distance_op, nopt, nopt, ndims);

    t1 = timer_rdtsc();
    pairwise_distance(q, nopt, x1, x2, distance_op, nopt, nopt, ndims);
    t2 = timer_rdtsc();
    printf("TIME: %.6lf\n", (double)(t2 - t1) / getHz());
    fflush(stdout);

    // ofstream file;
    // file.open("D.bin", ios::out|ios::binary);
    // if (file) {
    //   tfloat* tdistance_op = (tfloat*)_mm_malloc( nopt*nopt*sizeof(tfloat),
    //   ALIGN_FACTOR); q.memcpy(tdistance_op, distance_op,
    //   nopt*nopt*sizeof(tfloat)); q.wait();

    //   file.write(reinterpret_cast<char *>(tdistance_op),
    //   nopt*nopt*sizeof(tfloat)); file.close(); _mm_free(tdistance_op);
    // } else {
    //   std::cout << "Unable to open output file.\n";
    // }

    /* Deallocate arrays */
    FreeData(q, x1, x2, distance_op);

    return 0;
}
