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

void InitData( int nopt, tfloat* *x1, tfloat* *x2, tfloat* distance_op )
{
  tfloat *tx1, *tx2;
  int i;

  /* Allocate aligned memory */
  tx1 = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
  tx2 = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);

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
        #pragma omp for simd
        for ( i = 0; i < nopt; i++ )
        {
            tx1[i] = RandRange( XL, XH, &seed );
            tx2[i] = RandRange( XL, XH, &seed );
        }
    }

    *x1 = tx1;
    *x2 = tx2;
    *distance_op = 0;
}

/* Deallocate arrays */
void FreeData( tfloat *x1, tfloat *x2 )
{
    /* Free memory */
    _mm_free(x1);
    _mm_free(x2);
}
