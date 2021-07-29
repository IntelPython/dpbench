/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "constants_header.h"
#include "rdtsc.h"

int main(int argc, char * argv[])
{
    size_t nopt = 1 << 10;
    int repeat = 1;
    struct point *x1, *x2;
    tfloat* distance_op;

    clock_t t1 = 0, t2 = 0;

    int STEPS = 5;

    /* Read nopt number of options parameter from command line */
    if (argc < 2)
    {
        printf("Usage: expect STEPS input integer parameter, defaulting to %d\n", STEPS);
    }
    else
    {
        sscanf(argv[1], "%d", &STEPS);
	if (argc >= 3) {
	  sscanf(argv[2], "%lu", &nopt);
	}
	if (argc >= 4) {
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
    
    int i, j;
    for(i = 0; i < STEPS; i++) {
    
      /* Allocate arrays, generate input data */
      InitData( nopt, &x1, &x2, &distance_op );
        
      /* Warm up cycle */
      for(j = 0; j < 1; j++) {
	pairwise_distance( nopt, x1, x2, distance_op );
      }

      /* Compute call and put prices using compiler math libraries */
      printf("Pairwise Distance: Native-C-SVML: Size: %lu MOPS: ", nopt);
	
      t1 = timer_rdtsc();
      for(j = 0; j < repeat; j++) {
	pairwise_distance( nopt, x1, x2, distance_op );
      }
      t2 = timer_rdtsc();
      printf("%.6lf\n", (2.0 * nopt * 100 / 1e6)/((double) (t2 - t1) / getHz()));
      fflush(stdout);
      fprintf(fptr, "%lu,%.6lf\n",nopt,(2.0 * nopt * 100 )/((double) (t2 - t1) / getHz()));
      fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));
      /**************************/

#if 1//PRINT_RESULT
      tfloat total_distances = 0.0;
      for (size_t i = 0; i < nopt; i++) {
	for (size_t j = 0; j < nopt; j++) {
	  total_distances += distance_op[i*nopt + j];
	}
      }
      printf("Total distance = %lf\n", total_distances);
#endif
      /**************************/

      /* Deallocate arrays */
      FreeData( x1, x2 );

      nopt = nopt * 2;
      if (repeat > 2) repeat -= 2;
    }
    fclose(fptr);
    fclose(fptr1);
    return 0;
}
