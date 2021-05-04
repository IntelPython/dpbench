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

#include "euro_opt.h"

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
void InitData( size_t nopt, tfloat* *s0, tfloat* *x, tfloat* *t,
                   tfloat* *vcall_compiler, tfloat* *vput_compiler,
                   tfloat* *vcall_mkl, tfloat* *vput_mkl
             )
{
    tfloat *ts0, *tx, *tt, *tvcall_compiler, *tvput_compiler, *tvcall_mkl, *tvput_mkl;
    size_t i;

    /* Allocate aligned memory */
    ts0             = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
    tx              = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
    tt              = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
    tvcall_compiler = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
    tvput_compiler  = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
    tvcall_mkl      = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
    tvput_mkl       = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);

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
void FreeData( tfloat *s0, tfloat *x, tfloat *t,
                   tfloat *vcall_compiler, tfloat *vput_compiler,
                   tfloat *vcall_mkl, tfloat *vput_mkl
             )
{
    /* Free memory */
    _mm_free(s0);
    _mm_free(x);
    _mm_free(t);
    _mm_free(vcall_compiler);
    _mm_free(vput_compiler);
    _mm_free(vcall_mkl);
    _mm_free(vput_mkl);
}
