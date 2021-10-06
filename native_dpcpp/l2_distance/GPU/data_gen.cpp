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
#include <fstream>

#include "constants_header.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace std;

void InitData( queue* q, size_t nopt, tfloat* *x1, tfloat* *x2, tfloat* distance_op )
{
    tfloat *tx1, *tx2;

    /* Allocate aligned memory */
    tx1 = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
    tx2 = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);

    if ( (tx1 == NULL) || (tx2 == NULL) )
    {
        printf("Memory allocation failure\n");
        exit(-1);
    }

    ifstream file;
    file.open("x_data.bin", ios::in|ios::binary);
    if (file) {
      file.read(reinterpret_cast<char *>(tx1), nopt*sizeof(tfloat));
      file.close();
    } else {
      std::cout << "Input file not found.\n";
      exit(0);
    }

    file.open("y_data.bin", ios::in|ios::binary);
    if (file) {
      file.read(reinterpret_cast<char *>(tx2), nopt*sizeof(tfloat));
      file.close();
    } else {
      std::cout << "Input file not found.\n";
      exit(0);
    }

    *x1 = tx1;
    *x2 = tx2;
    *distance_op = 0.0;
}

/* Deallocate arrays */
void FreeData( queue* q, tfloat *x1, tfloat *x2 )
{
    /* Free memory */
    _mm_free(x1);
    _mm_free(x2);
}
