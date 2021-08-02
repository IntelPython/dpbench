/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

#include "data_gen.h"
#include "rdtsc.h"

#include "point.h"
#include "kmeans.h"

#define NUMBER_OF_CENTROIDS 10

int main(int argc, char * argv[])
{
  size_t nopt = 1 << 13;
    int repeat = 100;
    Point* points;
    Centroid* centroids;

    clock_t t1 = 0, t2 = 0;

    int STEPS = 3;

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

    queue *q = nullptr;

    try {
      q = new queue{cpu_selector()};
    } catch (runtime_error &re) {
      std::cerr << "No GPU device found\n";
      exit(1);
    }    
    
    int i, j;
    for(i = 0; i < STEPS; i++) {
    
      /* Allocate arrays, generate input data */
      InitData( nopt, NUMBER_OF_CENTROIDS, &points, &centroids );

      /* Warm up cycle */
      for(j = 0; j < 1; j++) {
	runKmeans(q, points, centroids, nopt, NUMBER_OF_CENTROIDS);
      }

      /* Compute call and put prices using compiler math libraries */
      printf("Kmeans: Native-C-SVML: Size: %lu MOPS: ", nopt);

      t1 = timer_rdtsc();
      for(j = 0; j < repeat; j++) {
	runKmeans(q, points, centroids, nopt, NUMBER_OF_CENTROIDS);
      }
      t2 = timer_rdtsc();
      
      printf("%.6lf\n", (2.0 * nopt * repeat / 1e6)/((double) (t2 - t1) / getHz()));
      printf("%lu ,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));
      fflush(stdout);
      fprintf(fptr, "%lu,%.6lf\n",nopt,(2.0 * nopt * 100 )/((double) (t2 - t1) / getHz()));
      fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

      /**************************/
      printCentroids (centroids, NUMBER_OF_CENTROIDS);

      /* Deallocate arrays */
      FreeData( points, centroids );

      nopt = nopt * 2;
      repeat -= 2;
    }
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
