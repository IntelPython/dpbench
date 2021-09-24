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

using namespace cl::sycl;

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
    double r;
    drand48_r(seed, &r);
    return r*(b-a) + a;
}

void InitData( queue* q, size_t nopt, struct point* *x1, struct point* *x2, tfloat** distance_op )
{
  struct point *tx1, *tx2;
  size_t i;

  /* Allocate aligned memory */
  tx1 = (struct point*)malloc_shared( nopt * sizeof(struct point), *q);
  tx2 = (struct point*)malloc_shared( nopt * sizeof(struct point), *q);

  if ( (tx1 == NULL) || (tx2 == NULL) )
    {
      printf("Memory allocation failure\n");
      exit(-1);
    }

  /* NUMA-friendly data init */
#pragma omp parallel
  {
    struct drand48_data seed;
    srand48_r(omp_get_thread_num()+SEED, &seed);
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

  tfloat* distance = (tfloat*)malloc_shared( nopt * nopt * sizeof(tfloat), *q);
  *distance_op = distance;

}

/* Deallocate arrays */
void FreeData( queue* q, struct point *x1, struct point *x2 )
{
    /* Free memory */
  free(x1, q->get_context());
  free(x2, q->get_context());
}
