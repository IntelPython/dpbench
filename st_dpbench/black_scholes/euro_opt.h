/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __EURO_OPT_BENCH_H
#define __EURO_OPT_BENCH_H

#include <CL/sycl.hpp>

#ifdef __DO_FLOAT__
typedef float tfloat;
#else
typedef double tfloat;
#endif

#define ALIGN_FACTOR 64

#define RISK_FREE 0.1f
#define VOLATILITY 0.2f

void InitData(sycl::queue &q,
              size_t nopt,
              tfloat **s0,
              tfloat **x,
              tfloat **t,
              tfloat **vcall_compiler,
              tfloat **vput_compiler);

void FreeData(sycl::queue &q,
              tfloat *s0,
              tfloat *x,
              tfloat *t,
              tfloat *vcall_compiler,
              tfloat *vput_compiler);

void BlackScholesFormula_Compiler(size_t nopt,
                                  sycl::queue &q,
                                  tfloat r,
                                  tfloat sig,
                                  tfloat *s0,
                                  tfloat *x,
                                  tfloat *t,
                                  tfloat *vcall,
                                  tfloat *vput);

#endif // #ifndef __EURO_OPT_BENCH_H
