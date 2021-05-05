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

#include "constants_header.h"

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
    double r;
    drand48_r(seed, &r);
    return r*(b-a) + a;
}

void InitData( size_t nopt, int ncentroids, Point** points, Centroid** centroids )
{
  Point *pts;
  Centroid *cents;
  int i;
  
  /* Allocate aligned memory */
  pts = (Point*)_mm_malloc( nopt * sizeof(Point), ALIGN_FACTOR);
  cents = (Centroid*)_mm_malloc( ncentroids * sizeof(Centroid), ALIGN_FACTOR);

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
void FreeData( Point *pts, Centroid * cents)
{
    /* Free memory */
    _mm_free(pts);
    _mm_free(cents);
}
