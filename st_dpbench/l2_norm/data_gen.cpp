/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#define _XOPEN_SOURCE
#define _DEFAULT_SOURCE
#include <fstream>
#include <ia32intrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "constants_header.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace std;

void InitData(queue &q,
              size_t nopt,
              size_t ndims,
              tfloat **a,
              tfloat **distance_op)
{
    tfloat *ta =
        (tfloat *)_mm_malloc(nopt * ndims * sizeof(tfloat), ALIGN_FACTOR);

    if (ta == NULL) {
        printf("Memory allocation failure\n");
        exit(-1);
    }

    ifstream file;
    file.open("a_data.bin", ios::in | ios::binary);
    if (file) {
        file.read(reinterpret_cast<char *>(ta), nopt * ndims * sizeof(tfloat));
        file.close();
    }
    else {
        std::cout << "Input file not found.\n";
        exit(0);
    }

    tfloat *d_ta = (tfloat *)malloc_device(nopt * ndims * sizeof(tfloat), q);
    tfloat *distance = (tfloat *)malloc_device(nopt * sizeof(tfloat), q);

    // copy data host to device
    q.memcpy(d_ta, ta, nopt * ndims * sizeof(tfloat));

    q.wait();

    *a = d_ta;
    *distance_op = distance;

    /* Free memory */
    _mm_free(ta);
}

/* Deallocate arrays */
void FreeData(queue &q, tfloat *a, tfloat *d)
{
    /* Free memory */
    free(a, q.get_context());
    free(d, q.get_context());
}
