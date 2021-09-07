/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <CL/sycl.hpp>

#include "euro_opt.h"
#include "rdtsc.h"

using namespace std;
using namespace cl::sycl;

int main(int argc, char * argv[])
{
    size_t nopt = 32768;
    int repeat = 1;
    tfloat *s0, *x, *t, *vcall_compiler, *vput_compiler;

    clock_t t1 = 0, t2 = 0;

    bool test = false;

    /* Read nopt number of options parameter from command line */
    if (argc >= 2) {
      sscanf(argv[1], "%lu", &nopt);
    }
    if (argc >= 3) {
      sscanf(argv[2], "%d", &repeat);
    }
    if (argc == 4) {
      char test_str[] = "-t";
      if (strcmp(test_str, argv[3]) == 0) {
	test = true;
      }
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "a");
    if(fptr1 == NULL) {
      printf("Error!");   
      exit(1);
    }

    queue *q = nullptr;

    try {
      q = new queue{gpu_selector()};
    } catch (sycl::exception &re) {
      std::cerr << "No GPU device found\n";
      exit(1);
    }
    
    /* Allocate arrays, generate input data */
    InitData(q, nopt, &s0, &x, &t, &vcall_compiler, &vput_compiler );

    /* Warm up cycle */
    BlackScholesFormula_Compiler( nopt, q, RISK_FREE, VOLATILITY, s0, x, t, vcall_compiler, vput_compiler );

    /* Compute call and put prices using compiler math libraries */
    printf("ERF: Native-C-SVML: Size: %lu MOPS: ", nopt);

    t1 = timer_rdtsc();
    for(int j = 0; j < repeat; j++) {
      BlackScholesFormula_Compiler( nopt, q, RISK_FREE, VOLATILITY, s0, x, t, vcall_compiler, vput_compiler );
    }
    t2 = timer_rdtsc();

    printf("%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));
    fflush(stdout);
    fprintf(fptr1, "%lu,%.6lf\n",nopt,((double) (t2 - t1) / getHz()));

    if (test) {
      ofstream file;
      file.open("call.bin", ios::out|ios::binary);
      if (file) {
	file.write(reinterpret_cast<char *>(vcall_compiler), nopt*sizeof(tfloat));
	file.close();
      } else {
	std::cout << "Unable to open output file.\n";
      }

      file.open("put.bin", ios::out|ios::binary);
      if (file) {
	file.write(reinterpret_cast<char *>(vput_compiler), nopt*sizeof(tfloat));
	file.close();
      } else {
	std::cout << "Unable to open output file.\n";
      }
    }
    
    /* Deallocate arrays */
    FreeData( q, s0, x, t, vcall_compiler, vput_compiler );

    fclose(fptr1);

    return 0;
}
