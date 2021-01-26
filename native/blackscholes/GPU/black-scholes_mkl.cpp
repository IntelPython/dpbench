/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <omp.h>
#include <mkl.h>
#include "euro_opt.h"

#ifdef __DO_FLOAT__
#   define VDIV(n,a,b,r)   vsDiv(n,a,b,r)
#   define VLOG(n,a,r)     vsLn(n,a,r)
#   define VEXP(n,a,r)     vsExp(n,a,r)
#   define VINVSQRT(n,a,r) vsInvSqrt(n,a,r)
#   define VERF(n,a,r)     vsErf(n,a,r)

#   define QUARTER         0.25f
#   define HALF            0.5f
#   define TWO             2.0f
#else
#   define VDIV(n,a,b,r)   vdDiv(n,a,b,r)
#   define VLOG(n,a,r)     vdLn(n,a,r)
#   define VEXP(n,a,r)     vdExp(n,a,r)
#   define VINVSQRT(n,a,r) vdInvSqrt(n,a,r)
#   define VERF(n,a,r)     vdErf(n,a,r)

#   define QUARTER         0.25
#   define HALF            0.5
#   define TWO             2.0
#endif

#if defined _VML_ACCURACY_EP_
#   define VML_ACC VML_EP
#elif defined _VML_ACCURACY_LA_
#   define VML_ACC VML_LA
#elif defined _VML_ACCURACY_HA_
#   define VML_ACC VML_HA
#else
#   error: _VML_ACCURACY_HA_/LA/EP should be defined in makefile
#endif

/* Set the reusable buffer for intermediate results */
#if !defined NBUF
#   define NBUF            1024
#endif

/*
// This function computes the Black-Scholes formula.
// Input parameters:
//     nopt - length of arrays
//     s0   - initial price
//     x    - strike price
//     t    - maturity
//
//     Implementation assumes fixed constant parameters
//     r    - risk-neutral rate
//     sig  - volatility
//
// Output arrays for call and put prices:
//     vcall, vput
//
// Note: the restrict keyword here tells the compiler
//       that none of the arrays overlap in memory.
//
// Note: the implementation assumes nopt is a multiple of NBUF
*/
void BlackScholesFormula_MKL( int nopt,
    tfloat r, tfloat sig, tfloat * restrict s0, tfloat * restrict x,
    tfloat * restrict t, tfloat * restrict vcall, tfloat * restrict vput )
{
    int i;
    tfloat mr = -r;
    tfloat sig_sig_two = sig * sig * TWO;

    #pragma omp parallel for                                 \
        shared(s0, x, t, vcall, vput, mr, sig_sig_two, nopt) \
        default(none)
    for ( i = 0; i < nopt; i+= NBUF )
    {
        int j;
        tfloat *a, *b, *c, *y, *z, *e;
        tfloat *d1, *d2, *w1, *w2;
        __declspec(align(ALIGN_FACTOR)) tfloat Buffer[NBUF*4];
        // This computes vector length for the last iteration of the loop
        // in case nopt is not exact multiple of NBUF
        #define MY_MIN(x, y) ((x) < (y)) ? (x) : (y)
        int nbuf = MY_MIN(NBUF, nopt - i);

        a      = Buffer + NBUF*0;          w1 = a; d1 = w1;
        c      = Buffer + NBUF*1;          w2 = c; d2 = w2;
        b      = Buffer + NBUF*2; e = b;
        z      = Buffer + NBUF*3; y = z;


        // Must set VML accuracy in each thread
        vmlSetMode( VML_ACC );

        VDIV(nbuf, s0 + i, x + i, a);
        VLOG(nbuf, a, a);

        #pragma vector vectorlength(512)
        for ( j = 0; j < nbuf; j++ )
        {
            b[j] = t[i + j] * mr;
            a[j] = a[j] - b[j];
            z[j] = t[i + j] * sig_sig_two;
            c[j] = QUARTER * z[j];
        }

        VINVSQRT(nbuf, z, y);
        VEXP(nbuf, b, e);

        #pragma vector vectorlength(512)
        for ( j = 0; j < nbuf; j++ )
        {
            tfloat aj = a[j];
            tfloat cj = c[j];
            w1[j] = ( aj + cj ) * y[j];
            w2[j] = ( aj - cj ) * y[j];
        }

        VERF(nbuf, w1, d1);
        VERF(nbuf, w2, d2);

        #pragma vector vectorlength(512)
        for ( j = 0; j < nbuf; j++ )
        {
            tfloat d1j = HALF + HALF*d1[j];
            tfloat d2j = HALF + HALF*d2[j];
			tfloat ej = e[j];
			tfloat s0j = s0[i+j];
			tfloat xj = x[i+j];
			tfloat vcallij =s0j*d1j - xj*ej*d2j;
            vcall[i+j] = vcallij;
            vput[i+j]  = vcallij - s0j + xj*ej;
        }
#if 0 
        for ( j = 0; j < nbuf; j++ )
        {
            d1[j] = HALF + HALF*d1[j];
            d2[j] = HALF + HALF*d2[j];
            vcall[i+j] = s0[i+j]*d1[j] - x[i+j]*e[j]*d2[j];
            vput[i+j]  = vcall[i+j] - s0[i+j] + x[i+j]*e[j];
        }
#endif
    }
}
