/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __EURO_OPT_BENCH_H
#define __EURO_OPT_BENCH_H

#ifdef __DO_FLOAT__
    typedef float tfloat;
#else
    typedef double  tfloat;
#endif

#define ALIGN_FACTOR 64

#define SEED 7777777

#define S0L     10.0f
#define S0H     50.0f
#define XL      10.0f
#define XH      50.0f

#define TL      1.0f
#define TH      2.0f

#define RISK_FREE  0.1f
#define VOLATILITY 0.2f

void InitData( size_t nopt, tfloat* *s0, tfloat* *x, tfloat* *t,
                   tfloat* *vcall_compiler, tfloat* *vput_compiler
             );

void FreeData( tfloat *s0, tfloat *x, tfloat *t,
                   tfloat *vcall_compiler, tfloat *vput_compiler
             );

void BlackScholesNaive( int nopt, tfloat r, tfloat sig, const tfloat so[],
    const tfloat x[], const tfloat t[], tfloat vcall[], tfloat vput[] );

void BlackScholesFormula_Compiler( size_t nopt,
    tfloat r, tfloat sig, tfloat *  s0, tfloat *  x,
    tfloat *  t, tfloat *  vcall, tfloat *  vput );

void BlackScholesFormula_MKL( int nopt,
    tfloat r, tfloat sig, tfloat *  s0, tfloat *  x,
    tfloat *  t, tfloat *  vcall, tfloat *  vput );

void BlackScholesFormula_CND_TBB( int nopt,
    tfloat r, tfloat sig, tfloat *  s0, tfloat *  x,
    tfloat *  t, tfloat *  vcall, tfloat *  vput );

void BlackScholesFormula_CND( int nopt,
    tfloat r, tfloat sig, tfloat *  s0, tfloat *  x,
    tfloat *  t, tfloat *  vcall, tfloat *  vput );

#endif // #ifndef __EURO_OPT_BENCH_H
