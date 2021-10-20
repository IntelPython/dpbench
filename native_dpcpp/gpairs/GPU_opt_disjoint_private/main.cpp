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
    size_t nopt = 1024;
    int repeat = 1;

    clock_t t1 = 0, t2 = 0;

    bool test = false;

    /* Read nopt number of options parameter from command line */
    if (argc >= 2) {
      sscanf(argv[1], "%lu", &nopt);
    }
    if (argc >= 3) {
      sscanf(argv[2], "%d", &repeat);
    }
    if (argc == 4) {
      char test_str[] = "-t";
      if (strcmp(test_str, argv[3]) == 0) {
	test = true;
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
    } catch (sycl::exception &re) {
      std::cerr << "No GPU device found\n";
      exit(1);
    }

    tfloat *x1, *y1, *z1, *w1, *x2, *y2, *z2, *w2, *rbins, *results_test;

    /* Allocate arrays, generate input data */
    InitData( q, nopt, &x1, &y1, &z1, &w1, &x2, &y2, &z2, &w2, &rbins, &results_test);

    /* Warm up cycle */
    call_gpairs( q, nopt, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_test);

    ResetResult(q, results_test);

    t1 = timer_rdtsc();
    for(int j = 0; j < repeat; j++) {
      call_gpairs( q, nopt, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_test );
    }
    t2 = timer_rdtsc();

    printf("%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));
    fflush(stdout);
    fprintf(fptr, "%lu,%.6lf\n",nopt,(2.0 * nopt * repeat )/((double) (t2 - t1) / getHz()));
    fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

    if (test) {
      ofstream file;
      file.open("result.bin", ios::out|ios::binary);
      if (file) {
    	file.write(reinterpret_cast<char *>(results_test), (DEFAULT_NBINS-1)*sizeof(tfloat));
    	file.close();
      } else {
    	std::cout << "Unable to open output file.\n";
      }
    }

#if 0 //print result
    for (size_t i = 0; i < (DEFAULT_NBINS-1); i++) {
      std::cout << results_test[i] << std::endl;
    }
#endif

    /* Deallocate arrays */
    FreeData( q, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_test );

    fclose(fptr);
    fclose(fptr1);

    return 0;
}
