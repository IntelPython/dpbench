/*
0;136;0c * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "constants_header.h"
#include "rdtsc.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main(int argc, char * argv[])
{
    size_t nopt = 1 << 16;
    int repeat = 100;
    tfloat *x1, *x2, distance_op;

    clock_t t1 = 0, t2 = 0;

    int STEPS = 10;

    /* Read nopt number of options parameter from command line */
    if (argc < 2)
    {
        printf("Usage: expect STEPS input integer parameter, defaulting to %d\n", STEPS);
    }
    else
    {
        sscanf(argv[1], "%d", &STEPS);

    if (argc == 3) {
      sscanf(argv[2], "%lu", &nopt);
    }

    }

    FILE *fptr;
    fptr = fopen("perf_output.csv", "a");
    if(fptr == NULL) {
      printf("Error!");
      exit(1);
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "a");
    if(fptr1 == NULL) {
      printf("Error!");
      exit(1);
    }

    queue *q = nullptr;

    try {
      q = new queue{gpu_selector()};
    } catch (runtime_error &re) {
      std::cerr << "No GPU device found\n";
      exit(1);
    }

    int i, j;
    for(i = 0; i < STEPS; i++) {

      /* Allocate arrays, generate input data */
      InitData( q, nopt, &x1, &x2, &distance_op );

      /* Warm up cycle */
      for(j = 0; j < 1; j++) {
      	l2_distance( q, nopt, x1, x2, &distance_op );
      }

      /* Compute call and put prices using compiler math libraries */
      printf("L2 Distance: Native-C-SVML: Size: %lu MOPS: ", nopt);

      t1 = timer_rdtsc();
      for(j = 0; j < repeat; j++) {
      	l2_distance( q, nopt, x1, x2, &distance_op );
      }
      t2 = timer_rdtsc();
      printf("%.6lf\n", (2.0 * nopt * 100 / 1e6)/((double) (t2 - t1) / getHz()));
      fflush(stdout);
      fprintf(fptr, "%lu,%.6lf\n",nopt,(2.0 * nopt * 100 )/((double) (t2 - t1) / getHz()));
      fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

      /**************************/

#if 0
      for (size_t k = 0; k < nopt; k++) {
	printf ("x1 : %lf\n", x1[k]);
	printf ("x2 : %lf\n", x2[k]);
      }
#endif
      printf("output distance = %lf\n", distance_op);

      /**************************/

      /* Deallocate arrays */
      FreeData( q, x1, x2 );

      nopt = nopt * 2;
      if (repeat > 2) repeat -= 2;
    }
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
