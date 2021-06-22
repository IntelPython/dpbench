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

using namespace cl::sycl;

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
    double r;
    drand48_r(seed, &r);
    return r*(b-a) + a;
}

void InitData( queue *q, size_t nopt, int ncentroids, Point** points, Centroid** centroids )
{
  Point *pts;
  Centroid *cents;
  int i;
  
  /* Allocate aligned memory */
  pts = (Point*)malloc_shared( nopt * sizeof(Point), *q);
  cents = (Centroid*)malloc_shared( ncentroids * sizeof(Centroid), *q);

  if ( (pts == NULL) || (cents == NULL) )
    {
      printf("Memory allocation failure\n");
      exit(-1);
    }

  /* NUMA-friendly data init */
#pragma omp parallel
  {
    struct drand48_data seed;
    srand48_r(omp_get_thread_num()+SEED, &seed);
#pragma omp for simd
    for ( i = 0; i < nopt; i++ )
      {
	pts[i].x = RandRange( XL, XH, &seed );
	pts[i].y = RandRange( XL, XH, &seed );
      }
  }

  *points = pts;
  *centroids = cents;
}

/* Deallocate arrays */
void FreeData( queue *q, Point *pts, Centroid * cents)
{
  free(pts, q->get_context());
  free(cents, q->get_context());
}
