/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#define _XOPEN_SOURCE
#define _DEFAULT_SOURCE
#include <fstream>
#include <ia32intrin.h>
#include <stdio.h>
#include <stdlib.h>

#include "constants_header.h"

using namespace std;

using namespace cl::sycl;

template <typename T>
void ReadInputFromBinFile(char const *filename,
                          char *data_ptr,
                          size_t data_size)
{
    ifstream file;
    file.open(filename, ios::in | ios::binary);
    if (file) {
        file.read(data_ptr, data_size * sizeof(T));
        file.close();
    }
    else {
        std::cout << "Input file - " << filename << " not found.\n";
        exit(0);
    }
}

void WriteOutputToTextFile(char const *filename,
                           tfloat *data_ptr,
                           size_t data_size)
{
    ofstream file;
    file.open(filename, ios::out);
    if (file) {
        for (size_t i = 0; i < data_size; i++) {
            file << *data_ptr << std::endl;
        }
        file.close();
    }
    else {
        std::cout << "Input file - " << filename << " not found.\n";
        exit(0);
    }
}

void InitData(queue &q,
              size_t nopt,
              size_t ndims,
              tfloat **x1,
              tfloat **x2,
              tfloat **distance_op)
{
    tfloat *tx1, *tx2;

    /* Allocate aligned memory */
    tx1 = (tfloat *)_mm_malloc(nopt * ndims * sizeof(tfloat), ALIGN_FACTOR);
    tx2 = (tfloat *)_mm_malloc(nopt * ndims * sizeof(tfloat), ALIGN_FACTOR);

    if ((tx1 == NULL) || (tx2 == NULL)) {
        printf("Memory allocation failure\n");
        exit(-1);
    }

    ReadInputFromBinFile<tfloat>("X.bin", reinterpret_cast<char *>(tx1),
                                 nopt * ndims);
    ReadInputFromBinFile<tfloat>("Y.bin", reinterpret_cast<char *>(tx2),
                                 nopt * ndims);

    tfloat *d_tx1 = (tfloat *)malloc_device(nopt * ndims * sizeof(tfloat), q);
    tfloat *d_tx2 = (tfloat *)malloc_device(nopt * ndims * sizeof(tfloat), q);
    tfloat *distance = (tfloat *)malloc_device(nopt * nopt * sizeof(tfloat), q);

    q.memcpy(d_tx1, tx1, nopt * ndims * sizeof(tfloat));
    q.memcpy(d_tx2, tx2, nopt * ndims * sizeof(tfloat));

    q.wait();

    *x1 = d_tx1;
    *x2 = d_tx2;
    *distance_op = distance;

    /* Free memory */
    _mm_free(tx1);
    _mm_free(tx2);
}

/* Deallocate arrays */
void FreeData(queue &q, tfloat *x1, tfloat *x2, tfloat *distance)
{
    free(x1, q.get_context());
    free(x2, q.get_context());
    free(distance, q.get_context());
}
