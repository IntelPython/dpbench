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

#include <chrono>

using namespace cl::sycl;

int main(int argc, char * argv[])
{
  size_t nopt = 1 << 16;
  int repeat = 1;
  tfloat *x1, *x2, distance_op;

  clock_t t1 = 0, t2 = 0;

  int STEPS = 10;

  bool test = false;
  double time;

  /* Read nopt number of options parameter from command line */
  if (argc >= 2)
    {
      sscanf(argv[1], "%lu", &nopt);
    }
  if (argc >= 3)
    {
      sscanf(argv[2], "%d", &repeat);
    }
  if (argc == 4)
    {
      char test_str[] = "-t";
      if (strcmp(test_str, argv[3]) == 0)
	{
	  test = true;
	}
    }

  FILE *fptr1;
  fptr1 = fopen("runtimes.csv", "w");
  if(fptr1 == NULL) {
    printf("Error!");
    exit(1);
  }

  queue *q = nullptr;
  try
    {
      q = new queue{gpu_selector()};
    }
  catch (sycl::exception &re)
    {
      std::cerr << "No GPU device found\n";
      exit(1);
    }

  /* Allocate arrays, generate input data */
  InitData( q, nopt, &x1, &x2, &distance_op );

  /* Warm up cycle */
  l2_distance( q, nopt, x1, x2, &distance_op );

  /* Compute call and put prices using compiler math libraries */
  printf("L2 Distance: Native-C-SVML: Size: %lu MOPS: ", nopt);

  // t1 = timer_rdtsc();
  auto start = std::chrono::high_resolution_clock::now();
  for(auto j = 0; j < repeat; j++)
    {
      l2_distance( q, nopt, x1, x2, &distance_op);
    }
  // t2 = timer_rdtsc();
  auto end = std::chrono::high_resolution_clock::now();
  // double time = ((double)(t2 - t1) / getHz());

  std::chrono::duration<double> elapsed_seconds = end - start;
  time = elapsed_seconds.count() / repeat;

  printf("%.6lf\n", (2.0 * nopt * 100 / 1e6)/((double) (t2 - t1) / getHz()));

  printf("ERF: Native-C-VML: Size: %ld Time: %.6lf\n", nopt, time);

  fflush(stdout);
  // fprintf(fptr, "%lu,%.6lf\n",nopt,(2.0 * nopt * 100 )/((double) (t2 - t1) / getHz()));
  fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

  /**************************/

  //std::cout << "DISTANCE : " << distance_op << std::endl;

  if (test)
    {
      std::ofstream file;
      file.open("distance.bin", std::ios::out|std::ios::binary);
      if (file)
	{
	  file.write(reinterpret_cast<const char*>(&distance_op), sizeof(tfloat));
	  file.close();
	}
      else
	{
	  std::cout << "Unable to open output file.\n";
	}
    }

  fclose(fptr1);

  FreeData(q, x1, x2);

  return 0;
}
