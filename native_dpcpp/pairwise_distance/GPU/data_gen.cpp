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

void InitData( size_t nopt, struct point* *x1, struct point* *x2, tfloat** distance_op )
{
  struct point *tx1, *tx2;
  size_t i;
  
  /* Allocate aligned memory */
  tx1 = (struct point*)_mm_malloc( nopt * sizeof(struct point), ALIGN_FACTOR);
  tx2 = (struct point*)_mm_malloc( nopt * sizeof(struct point), ALIGN_FACTOR);

  if ( (tx1 == NULL) || (tx2 == NULL) )
    {
      printf("Memory allocation failure\n");
      exit(-1);
    }

  /* NUMA-friendly data init */
  //#pragma omp parallel
  {
    struct drand48_data seed;
    srand48_r(omp_get_thread_num()+SEED, &seed);
    //#pragma omp for simd
    for ( i = 0; i < nopt; i++ )
      {
	tx1[i].x = RandRange( XL, XH, &seed );
	tx1[i].y = RandRange( XL, XH, &seed );
	tx1[i].z = RandRange( XL, XH, &seed );
	tx2[i].x = RandRange( XL, XH, &seed );
	tx2[i].y = RandRange( XL, XH, &seed );
	tx2[i].z = RandRange( XL, XH, &seed );
      }
  }

  *x1 = tx1;
  *x2 = tx2;

  tfloat* distance = (tfloat*)_mm_malloc( nopt * nopt * sizeof(tfloat), ALIGN_FACTOR);
  *distance_op = distance;

}

/* Deallocate arrays */
void FreeData( struct point *x1, struct point *x2 )
{
    /* Free memory */
    _mm_free(x1);
    _mm_free(x2);
}
