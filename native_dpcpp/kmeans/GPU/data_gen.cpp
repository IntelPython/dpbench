/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#define _XOPEN_SOURCE
#define _DEFAULT_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <ia32intrin.h>

#include "data_gen.h"

using namespace std;
using namespace cl::sycl;

void WriteOutputToTextFile (char const * filename, Point* data_ptr, size_t data_size) {
    ofstream file;
    file.open(filename, ios::out);

    if (file) {
      for (size_t i = 0; i < data_size; i++) {
	file << data_ptr[i].x << "," << data_ptr[i].y << std::endl;
      }
      file.close();
    } else {
      std::cout << "Input file - " << filename << " not found.\n";
      exit(0);
    }
}

void InitData( queue *q, size_t nopt, int ncentroids, Point** points, Centroid** centroids )
{
  Point *pts;
  Centroid *cents;
  int i;

  /* Allocate aligned memory */
  pts = (Point*)_mm_malloc( nopt * sizeof(Point), ALIGN_FACTOR);
  cents = (Centroid*)_mm_malloc( ncentroids * sizeof(Centroid), ALIGN_FACTOR);

  if ( (pts == NULL) || (cents == NULL) ) {
    printf("Memory allocation failure\n");
    exit(-1);
  }

  ifstream file;
  file.open("X.bin", ios::in|ios::binary);

  for ( i = 0; i < nopt; i++ ) {
    file.read(reinterpret_cast<char*>(&pts[i].x), sizeof(tfloat));
    file.read(reinterpret_cast<char*>(&pts[i].y), sizeof(tfloat));
  }

  file.close();

  //WriteOutputToTextFile("X_dpcpp.txt", pts, nopt);
  *points = pts;
  *centroids = cents;
}

/* Deallocate arrays */
void FreeData( queue *q, Point *pts, Centroid * cents)
{
    /* Free memory */
    _mm_free(pts);
    _mm_free(cents);
}
