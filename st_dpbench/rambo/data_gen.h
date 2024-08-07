/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <fstream>
#include <ia32intrin.h>
#include <stdio.h>
#include <stdlib.h>

#define ALIGN_FACTOR 64

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

void InitData(queue &q,
              size_t nevts,
              size_t nout,
              double **C1,
              double **F1,
              double **Q1,
              double **output)
{
    double *tC1, *tF1, *tQ1;

    /* Allocate aligned memory */
    tC1 = (double *)_mm_malloc(nevts * nout * sizeof(double), ALIGN_FACTOR);
    tF1 = (double *)_mm_malloc(nevts * nout * sizeof(double), ALIGN_FACTOR);
    tQ1 = (double *)_mm_malloc(nevts * nout * sizeof(double), ALIGN_FACTOR);

    if ((tC1 == NULL) || (tF1 == NULL) || (tQ1 == NULL)) {
        printf("Memory allocation failure\n");
        exit(-1);
    }

    ReadInputFromBinFile<double>("C1.bin", reinterpret_cast<char *>(tC1),
                                 nevts * nout);
    ReadInputFromBinFile<double>("F1.bin", reinterpret_cast<char *>(tF1),
                                 nevts * nout);
    ReadInputFromBinFile<double>("Q1.bin", reinterpret_cast<char *>(tQ1),
                                 nevts * nout);

    double *d_tC1 = (double *)malloc_device(nevts * nout * sizeof(double), q);
    double *d_tF1 = (double *)malloc_device(nevts * nout * sizeof(double), q);
    double *d_tQ1 = (double *)malloc_device(nevts * nout * sizeof(double), q);
    double *d_output =
        (double *)malloc_device(nevts * nout * 4 * sizeof(double), q);

    q.memcpy(d_tC1, tC1, nevts * nout * sizeof(double));
    q.memcpy(d_tF1, tF1, nevts * nout * sizeof(double));
    q.memcpy(d_tQ1, tQ1, nevts * nout * sizeof(double));

    q.wait();

    *C1 = d_tC1;
    *F1 = d_tF1;
    *Q1 = d_tQ1;
    *output = d_output;

    /* Free memory */
    _mm_free(tC1);
    _mm_free(tF1);
    _mm_free(tQ1);
}

/* Deallocate arrays */
void FreeData(queue &q, double *C1, double *F1, double *Q1, double *output)
{
    free(C1, q.get_context());
    free(F1, q.get_context());
    free(Q1, q.get_context());
    free(output, q.get_context());
}
