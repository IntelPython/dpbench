/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#define _XOPEN_SOURCE
#define _DEFAULT_SOURCE 
#include <stdlib.h>
#include <stdio.h>
#include <ia32intrin.h>
#include <CL/sycl.hpp>
#include <omp.h>

#include "euro_opt.h"

using namespace cl::sycl;

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
    double r;
    drand48_r(seed, &r);
    return r*(b-a) + a;
}

/*
// This function allocates arrays to hold input and output parameters
// for the Black-Scholes formula.
//     nopt - length of arrays
// Random input parameters
//     s0   - initial price
//     x    - strike price
//     t    - maturity
// Output arrays for call and put prices
//     vcall_compiler, vcall_mkl
//     vput_compiler, vput_mkl
*/
void InitData( queue *q, size_t nopt, tfloat* *s0, tfloat* *x, tfloat* *t,
                   tfloat* *vcall_compiler, tfloat* *vput_compiler,
                   tfloat* *vcall_mkl, tfloat* *vput_mkl
             )
{
    tfloat *ts0, *tx, *tt, *tvcall_compiler, *tvput_compiler, *tvcall_mkl, *tvput_mkl;
    size_t i;

    /* Allocate aligned memory */
    ts0             = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    tx              = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    tt              = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    tvcall_compiler = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    tvput_compiler  = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    tvcall_mkl      = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    tvput_mkl       = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);

    if ( (ts0 == NULL) || (tx == NULL) || (tt == NULL) ||
         (tvcall_compiler == NULL) || (tvput_compiler == NULL) ||
         (tvcall_mkl == NULL) || (tvput_mkl == NULL) )
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
            ts0[i] = RandRange( S0L, S0H, &seed );
            tx[i]  = RandRange( XL, XH, &seed );
            tt[i]  = RandRange( TL, TH, &seed );

            tvcall_compiler[i] = 0.0;
            tvput_compiler[i]  = 0.0;
            tvcall_mkl[i] = 0.0;
            tvput_mkl[i]  = 0.0;
        }
    }

    *s0 = ts0;
    *x  = tx;
    *t  = tt;
    *vcall_compiler = tvcall_compiler;
    *vput_compiler  = tvput_compiler;
    *vcall_mkl = tvcall_mkl;
    *vput_mkl  = tvput_mkl;
}

/* Deallocate arrays */
void FreeData( queue* q, tfloat *s0, tfloat *x, tfloat *t,
                   tfloat *vcall_compiler, tfloat *vput_compiler,
                   tfloat *vcall_mkl, tfloat *vput_mkl
             )
{
    /* Free memory */
  free(s0, q->get_context());
  free(x, q->get_context());
  free(t, q->get_context());
  free(vcall_compiler, q->get_context());
  free(vput_compiler, q->get_context());
  free(vcall_mkl, q->get_context());
  free(vput_mkl, q->get_context());
}
