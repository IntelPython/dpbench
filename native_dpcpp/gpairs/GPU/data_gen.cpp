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
#include <cmath>

#include "euro_opt.h"

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
    double r;
    drand48_r(seed, &r);
    return r*(b-a) + a;
}

void InitRbins_Results(tfloat **rbins, tfloat **results_test) {
  const float DEFAULT_RMIN = 0.1;
  const int DEFAULT_RMAX = 50;

  *rbins = (tfloat*)_mm_malloc(DEFAULT_NBINS * sizeof(tfloat), ALIGN_FACTOR);
  //result = (tfloat*)_mm_malloc(DEFAULT_NBINS * sizeof(tfloat), ALIGN_FACTOR);
  *results_test = (tfloat*)_mm_malloc((DEFAULT_NBINS-1) * sizeof(tfloat), ALIGN_FACTOR);
  
  tfloat start = log10(DEFAULT_RMIN);
  tfloat stop = log10(DEFAULT_RMAX);
  
  tfloat curval = pow(10, start);
  tfloat baseval = pow(10, ((stop-start)/DEFAULT_NBINS));

  for (unsigned int i = 0; i < DEFAULT_NBINS; i++) {
    (*rbins)[i] = curval * curval;
    curval *= baseval;

    if (i != (DEFAULT_NBINS-1)){
      (*results_test)[i] = 0;
    }
  }

}

void InitData(size_t npoints, tfloat **x1, tfloat **y1, tfloat **z1, tfloat **w1,
	      tfloat **x2, tfloat **y2, tfloat **z2, tfloat **w2, tfloat **rbins, tfloat **results_test) {
  /* Allocate aligned memory */
  *x1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *y1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *z1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *w1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *x2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *y2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *z2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *w2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);

  if ( (*x1 == NULL) || (*y1 == NULL) || (*z1 == NULL) || (*w1 == NULL) ||
       (*x2 == NULL) || (*y2 == NULL) || (*z2 == NULL) || (*w2 == NULL)) {
    printf("Memory allocation failure\n");
    exit(-1);
  }

  size_t i;
  /* NUMA-friendly data init */
#pragma omp parallel
  {
    struct drand48_data seed;
    srand48_r(omp_get_thread_num()+SEED, &seed);
#pragma omp for simd
    for ( i = 0; i < npoints; i++ )
      {
	(*x1)[i] = RandRange( 0.0, LBOX, &seed );
	(*y1)[i] = RandRange( 0.0, LBOX, &seed );
	(*z1)[i] = RandRange( 0.0, LBOX, &seed );
	(*w1)[i] = RandRange( 0.0, TL, &seed );

	(*x2)[i] = RandRange( 0.0, LBOX, &seed );
	(*y2)[i] = RandRange( 0.0, LBOX, &seed );
	(*z2)[i] = RandRange( 0.0, LBOX, &seed );
	(*w2)[i] = RandRange( 0.0, TL, &seed );
      }
  }

  InitRbins_Results(rbins, results_test);
}

/* Deallocate arrays */
void FreeData( tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
	       tfloat *x2, tfloat *y2, tfloat *z2, tfloat *w2, tfloat *rbins, tfloat *results_test )
{
    /* Free memory */
    _mm_free(x1);
    _mm_free(y1);
    _mm_free(z1);
    _mm_free(w1);
    _mm_free(x2);
    _mm_free(y2);
    _mm_free(z2);
    _mm_free(w2);
    _mm_free(rbins);
    _mm_free(results_test);
}
