/*
  0;136;0c * Copyright (C) 2014-2015, 2018 Intel Corporation
  *
  * SPDX-License-Identifier: MIT
  */

#include "constants_header.h"
#include "rdtsc.h"
#include <CL/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cl::sycl;

int main(int argc, char *argv[])
{
    size_t nopt = 536870912;
    size_t ndims = 3;
    tfloat *a, *distance_op;

    clock_t t1 = 0, t2 = 0;

    queue q;

    /* Allocate arrays, generate input data */
    InitData(q, nopt, ndims, &a, &distance_op);

    /* Warm up cycle */
    l2_norm_impl(q, nopt, ndims, a, distance_op);

    t1 = timer_rdtsc();
    l2_norm_impl(q, nopt, ndims, a, distance_op);
    t2 = timer_rdtsc();

    printf("TIME: %.6lf\n", ((double)(t2 - t1) / getHz()));
    fflush(stdout);

    FreeData(q, a, distance_op);

    return 0;
}
