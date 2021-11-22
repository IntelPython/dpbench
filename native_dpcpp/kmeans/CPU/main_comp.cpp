/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <CL/sycl.hpp>

#include "rdtsc.h"
#include "data_gen.h"
#include "point.h"
#include "kmeans.h"

using namespace std;
using namespace cl::sycl;

#define NUMBER_OF_CENTROIDS 10

int main(int argc, char * argv[])
{
  size_t nopt = 1 << 13;
  int repeat = 1;
  Point* points;
  Centroid* centroids;

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
    q = new queue{cpu_selector()};
  } catch (sycl::exception &re) {
    std::cerr << "No GPU device found\n";
    exit(1);
  }

  /* Allocate arrays, generate input data */
  InitData( q, nopt, NUMBER_OF_CENTROIDS, &points, &centroids );

  /* Warm up cycle */
  runKmeans(q, points, centroids, nopt, NUMBER_OF_CENTROIDS);

  /* Compute call and put prices using compiler math libraries */
  printf("Kmeans: Native-C-SVML: Size: %lu MOPS: ", nopt);

  t1 = timer_rdtsc();
  for(int j = 0; j < repeat; j++) {
    runKmeans(q, points, centroids, nopt, NUMBER_OF_CENTROIDS);
  }
  t2 = timer_rdtsc();

  printf("%.6lf\n", (2.0 * nopt * repeat / 1e6)/((double) (t2 - t1) / getHz()));
  printf("%lu ,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));
  fflush(stdout);
  fprintf(fptr, "%lu,%.6lf\n",nopt,(2.0 * nopt * 100 )/((double) (t2 - t1) / getHz()));
  fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

  if (test) {
    Centroid* t_centroids = (Centroid*)_mm_malloc( NUMBER_OF_CENTROIDS * sizeof(Centroid), ALIGN_FACTOR);
    q->memcpy(t_centroids, centroids, NUMBER_OF_CENTROIDS * sizeof(Centroid));
    q->wait();

    ofstream file1, file2, file3;
    file1.open("arrayC.bin", ios::out|ios::binary|ios::app);
    file2.open("arrayCsum.bin", ios::out|ios::binary|ios::app);
    file3.open("arrayCnumpoint.bin", ios::out|ios::binary|ios::app);
    if (file1 && file2 && file3) {
      for (int i = 0; i < NUMBER_OF_CENTROIDS; i++) {
  	file1.write(reinterpret_cast<char *>(&t_centroids[i].x), sizeof(tfloat));
  	file1.write(reinterpret_cast<char *>(&t_centroids[i].y), sizeof(tfloat));

  	file2.write(reinterpret_cast<char *>(&t_centroids[i].x_sum), sizeof(tfloat));
  	file2.write(reinterpret_cast<char *>(&t_centroids[i].y_sum), sizeof(tfloat));

  	file3.write(reinterpret_cast<char *>(&t_centroids[i].num_points), sizeof(tint));
      }
      file1.close();
      file2.close();
      file3.close();
    } else {
      std::cout << "Unable to open output file.\n";
    }
  }

  /**************************/
  //printCentroids (centroids, NUMBER_OF_CENTROIDS);

  /* Deallocate arrays */
  FreeData( q, points, centroids );

  fclose(fptr);
  fclose(fptr1);

  return 0;
}
