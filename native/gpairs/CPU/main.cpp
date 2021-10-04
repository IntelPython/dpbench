/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "euro_opt.h"
#include "rdtsc.h"

int main(int argc, char * argv[])
{
    int nopt = 1024;
    int repeat = 1;
    int STEPS = 10;

    clock_t t1 = 0, t2 = 0;

    /* Read nopt number of options parameter from command line */
    if (argc < 2)
    {
        printf("Usage: expect STEPS input integer parameter, defaulting to %d\n", STEPS);
    }
    else
    {
        sscanf(argv[1], "%d", &STEPS);
	if (argc == 3) {
	  sscanf(argv[2], "%d", &nopt);
	}
	if (argc == 4) {
	  sscanf(argv[3], "%d", &repeat);
	}
    }

    FILE *fptr;
    fptr = fopen("perf_output.csv", "w");
    if(fptr == NULL) {
      printf("Error!");
      exit(1);
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "w");
    if(fptr1 == NULL) {
      printf("Error!");
      exit(1);
    }
    tfloat *x1, *y1, *z1, *w1, *x2, *y2, *z2, *w2, *rbins, *results_test;
    int i, j;
    for(i = 0; i < STEPS; i++) {
      /* Allocate arrays, generate input data */
      InitData( nopt, &x1, &y1, &z1, &w1, &x2, &y2, &z2, &w2, &rbins, &results_test);

      /* Warm up cycle */
      for(j = 0; j < 1; j++) {
	call_gpairs( nopt, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_test);
      }

      t1 = timer_rdtsc();
      for(j = 0; j < repeat; j++) {
	call_gpairs( nopt, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_test );
      }
      t2 = timer_rdtsc();
      printf("%d,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));
      fflush(stdout);
      fprintf(fptr, "%d,%.6lf\n",nopt,(2.0 * nopt * repeat )/((double) (t2 - t1) / getHz()));
      fprintf(fptr1, "%d,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

      /* Deallocate arrays */
      FreeData( x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_test );

      nopt = nopt * 2;
      if (repeat > 2) repeat -= 2;
    }

    fclose(fptr);
    fclose(fptr1);

    return 0;
}
