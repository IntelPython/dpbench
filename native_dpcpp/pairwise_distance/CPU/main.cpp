/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/sycl.hpp>
#include "constants_header.h"
#include "rdtsc.h"

using namespace cl::sycl;
using namespace std;

int main(int argc, char * argv[])
{
  size_t nopt = 1 << 10;
  int repeat = 1;
  struct point *x1, *x2;
  tfloat* distance_op;

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
  } catch (sycl::exception &re) {
    std::cerr << "No CPU device found\n";
    exit(1);
  }

  /* Allocate arrays, generate input data */
  InitData( q, nopt, &x1, &x2, &distance_op );

  /* Warm up cycle */
  pairwise_distance( q, nopt, x1, x2, distance_op );

  /* Compute call and put prices using compiler math libraries */
  printf("Pairwise Distance: Native-C-SVML: Size: %lu MOPS: ", nopt);

  t1 = timer_rdtsc();
  for(unsigned int j = 0; j < repeat; j++) {
    pairwise_distance( q, nopt, x1, x2, distance_op );
  }
  t2 = timer_rdtsc();
  printf("%.6lf\n", (2.0 * nopt * 100 / 1e6)/((double) (t2 - t1) / getHz()));
  fflush(stdout);
  fprintf(fptr, "%lu,%.6lf\n",nopt,(2.0 * nopt * 100 )/((double) (t2 - t1) / getHz()));
  fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

  if (test) {
    ofstream file;
    file.open("D.bin", ios::out|ios::binary);
    if (file) {
      file.write(reinterpret_cast<char *>(distance_op), nopt*nopt*sizeof(tfloat));
      file.close();
    } else {
      std::cout << "Unable to open output file.\n";
    }
  }

  /* Deallocate arrays */
  FreeData( q, x1, x2 );

  nopt = nopt * 2;

  fclose(fptr);
  fclose(fptr1);
  return 0;
}
