/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <omp.h>
#include <mathimf.h>
#include "euro_opt.h"

#ifdef __DO_FLOAT__
#   define EXP(x)      expf(x)
#   define LOG(x)      logf(x)
#   define SQRT(x)     sqrtf(x)
#   define ERF(x)      erff(x)
#   define INVSQRT(x)  1.0f/sqrtf(x)

#   define QUARTER     0.25f
#   define HALF        0.5f
#   define TWO         2.0f
#else
#   define EXP(x)      exp(x)
#   define LOG(x)      log(x)
#   define SQRT(x)     sqrt(x)
#   define ERF(x)      erf(x)
#   define INVSQRT(x)  1.0/sqrt(x)

#   define QUARTER     0.25
#   define HALF        0.5
#   define TWO         2.0
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
*/
void BlackScholesFormula_Compiler( size_t nopt,
    tfloat r, tfloat sig, tfloat *  s0, tfloat *  x,
    tfloat *  t, tfloat *  vcall, tfloat *  vput )
{
    size_t i;
    tfloat a, b, c, y, z, e;
    tfloat d1, d2, w1, w2;
    tfloat mr = -r;
    tfloat sig_sig_two = sig * sig * TWO;

#pragma omp target teams distribute \
  parallel for simd shared(s0, x, t, vcall, vput) private(a, b, z, c, e, y, d1, d2, w1, w2)
#pragma vector
    for ( i = 0; i < nopt; i++ )
    {
        a = LOG( s0[i] / x[i] );
        b = t[i] * mr;
        z = t[i] * sig_sig_two;
        
        c = QUARTER * z;
        e = EXP ( b );
        y = INVSQRT( z );
                             
        w1 = ( a - b + c ) * y;
        w2 = ( a - b - c ) * y;
        d1 = ERF( w1 );
        d2 = ERF( w2 );
        d1 = HALF + HALF*d1;
        d2 = HALF + HALF*d2;

        vcall[i] = s0[i]*d1 - x[i]*e*d2;
        vput[i]  = vcall[i] - s0[i] + x[i]*e;
    }
}

void BlackScholesNaive(
    size_t nopt, tfloat r, tfloat sig, const tfloat s0[], const tfloat x[],
    const tfloat t[], tfloat vcall[], tfloat vput[] )
{
    tfloat d1, d2, w1, w2;
    int i;

    for ( i=0; i<nopt; i++ )
    {
        d1 = ( LOG(s0[i]/x[i]) + (r + HALF*sig*sig)*t[i] ) /
             ( sig*SQRT(t[i]) );
        d2 = ( LOG(s0[i]/x[i]) + (r - HALF*sig*sig)*t[i] ) /
             ( sig*SQRT(t[i]) );

        w1 = HALF + HALF * ERF(d1 / SQRT(2));
        w2 = HALF + HALF * ERF(d2 / SQRT(2));

        vcall[i] = s0[i] * w1 - EXP(-r * t[i]) * x[i] * w2;
        vput[i] = EXP(-r*t[i])*x[i] * -w2 - s0[i] * -w1;
    }
}
