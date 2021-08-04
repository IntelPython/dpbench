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

using namespace cl::sycl;

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
    double r;
    drand48_r(seed, &r);
    return r*(b-a) + a;
}

void InitRbins_Results(queue *q, tfloat **rbins, tfloat **results_test) {
  const float DEFAULT_RMIN = 0.1;
  const int DEFAULT_RMAX = 50;

  *rbins = (tfloat*)malloc_shared(DEFAULT_NBINS * sizeof(tfloat), *q);
  *results_test = (tfloat*)malloc_shared((DEFAULT_NBINS-1) * sizeof(tfloat), *q);
  
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

void InitData(queue* q, size_t npoints, tfloat **x1, tfloat **y1, tfloat **z1, tfloat **w1,
	      tfloat **x2, tfloat **y2, tfloat **z2, tfloat **w2, tfloat **rbins, tfloat **results_test) {
  /* Allocate aligned memory */
  *x1 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);
  *y1 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);
  *z1 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);
  *w1 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);
  *x2 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);
  *y2 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);
  *z2 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);
  *w2 = (tfloat*)malloc_shared(npoints * sizeof(tfloat), *q);

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

  InitRbins_Results(q, rbins, results_test);
}

/* Deallocate arrays */
void FreeData( queue* q, tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
	       tfloat *x2, tfloat *y2, tfloat *z2, tfloat *w2, tfloat *rbins, tfloat *results_test )
{
    /* Free memory */
  free(x1, q->get_context());
  free(y1, q->get_context());
  free(z1, q->get_context());
  free(w1, q->get_context());
  free(x2, q->get_context());
  free(y2, q->get_context());
  free(z2, q->get_context());
  free(w2, q->get_context());
  free(rbins, q->get_context());
  free(results_test, q->get_context());
}
