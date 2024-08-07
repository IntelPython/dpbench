/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <CL/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "euro_opt.h"
#include "rdtsc.h"

using namespace std;
using namespace cl::sycl;

int main(int argc, char *argv[])
{
    size_t nopt = 268435456;
    tfloat *s0, *x, *t, *vcall_compiler, *vput_compiler;

    clock_t t1 = 0, t2 = 0;

    queue q;

    /* Allocate arrays, generate input data */
    InitData(q, nopt, &s0, &x, &t, &vcall_compiler, &vput_compiler);

    /* Warm up cycle */
    BlackScholesFormula_Compiler(nopt, q, RISK_FREE, VOLATILITY, s0, x, t,
                                 vcall_compiler, vput_compiler);

    t1 = timer_rdtsc();
    BlackScholesFormula_Compiler(nopt, q, RISK_FREE, VOLATILITY, s0, x, t,
                                 vcall_compiler, vput_compiler);
    t2 = timer_rdtsc();

    printf("TIME = %.6lf\n", ((double)(t2 - t1) / getHz()));
    fflush(stdout);

    /* Deallocate arrays */
    FreeData(q, s0, x, t, vcall_compiler, vput_compiler);

    return 0;
}
