/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "euro_opt.h"
#include "rdtsc.h"

int main(int argc, char * argv[])
{
    int nopt = 32768;
    int repeat = 1;
    tfloat *s0, *x, *t, *vcall_mkl, *vput_mkl, *vcall_compiler, *vput_compiler;

    clock_t t1 = 0, t2 = 0;

    int STEPS = 14;

    /* Read nopt number of options parameter from command line */
    if (argc < 2)
    {
        printf("Usage: expect STEPS input integer parameter, defaulting to %d\n", STEPS);
    }
    else
    {
        sscanf(argv[1], "%d", &STEPS);
	if (argc == 3) {
	  sscanf(argv[2], "%d", &nopt);
	}
    }

    FILE *fptr;
    fptr = fopen("perf_output.csv", "w");
    if(fptr == NULL) {
      printf("Error!");   
      exit(1);             
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "w");
    if(fptr1 == NULL) {
      printf("Error!");   
      exit(1);             
    }
    
    int i, j;
    for(i = 0; i < STEPS; i++) {
    
        /* Allocate arrays, generate input data */
        InitData( nopt, &s0, &x, &t, &vcall_compiler, &vput_compiler, &vcall_mkl, &vput_mkl );

        /* Warm up cycle */
        for(j = 0; j < 1; j++) {
#ifdef BLACK_SCHOLES_MKL
            BlackScholesFormula_MKL( nopt, RISK_FREE, VOLATILITY, s0, x, t, vcall_mkl, vput_mkl );
#else
            BlackScholesFormula_Compiler( nopt, RISK_FREE, VOLATILITY, s0, x, t, vcall_compiler, vput_compiler );
#endif
        }

#ifdef BLACK_SCHOLES_MKL
        /* Compute call and put prices using MKL VML functions */
        printf("ERF: Native-C-VML: Size: %d MOPS: ", nopt);
#else
        /* Compute call and put prices using compiler math libraries */
        printf("ERF: Native-C-SVML: Size: %d MOPS: ", nopt);
#endif
        t1 = timer_rdtsc();
        for(j = 0; j < repeat; j++) {
#ifdef BLACK_SCHOLES_MKL
            BlackScholesFormula_MKL( nopt, RISK_FREE, VOLATILITY, s0, x, t, vcall_mkl, vput_mkl );
#else
            BlackScholesFormula_Compiler( nopt, RISK_FREE, VOLATILITY, s0, x, t, vcall_compiler, vput_compiler );
#endif
        }
        t2 = timer_rdtsc();
        printf("%.6lf\n", (2.0 * nopt * repeat / 1e6)/((double) (t2 - t1) / getHz()));
        fflush(stdout);
	fprintf(fptr, "%d,%.6lf\n",nopt,(2.0 * nopt * repeat )/((double) (t2 - t1) / getHz()));
	fprintf(fptr1, "%d,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

        /* Deallocate arrays */
        FreeData( s0, x, t, vcall_compiler, vput_compiler, vcall_mkl, vput_mkl );

        nopt = nopt * 2;
	if (repeat > 2) repeat -= 2;
    }
    fclose(fptr);
    fclose(fptr1);

    /* Display a few computed values */
    // printf("call_compiler[0/%d]= %g\n", nopt, (double)(vcall_compiler[10]) );
    // printf("put_compiler[0/%d]= %g\n", nopt, (double)(vput_compiler[10]) );
    // printf("call_mkl[0/%d]= %g\n", nopt, (double)(vcall_mkl[10]) );
    // printf("put_mkl[0/%d]= %g\n", nopt, (double)(vput_mkl[10]) );

    return 0;
}
