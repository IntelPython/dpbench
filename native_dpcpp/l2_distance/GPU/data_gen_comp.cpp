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

#include "constants_header.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace std;


void InitData( queue* q, size_t nopt, tfloat* *x1, tfloat* *x2, tfloat* distance_op)
{
  tfloat *tx1, *tx2;

  /* Allocate aligned memory */

  tx1 = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
  tx2 = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);

  if ( (tx1 == NULL) || (tx2 == NULL) )
  {
      printf("Memory allocation failure\n");
      exit(-1);
  }

  ifstream file;
  file.open("x_data.bin", ios::in|ios::binary);
  if (file)
  {
        file.read(reinterpret_cast<char *>(tx1), nopt*sizeof(tfloat));
        file.close();
  } else
  {
        std::cout << "Input file not found.\n";
        exit(0);
  }

  file.open("y_data.bin", ios::in|ios::binary);
  if (file)
  {
        file.read(reinterpret_cast<char *>(tx2), nopt*sizeof(tfloat));
        file.close();
  } else
  {
        std::cout << "Input file not found.\n";
        exit(0);
  }

  tfloat *d_tx1, *d_tx2;
  d_tx1 = (tfloat*)malloc_device(nopt * sizeof(tfloat), *q);
  d_tx2 = (tfloat*)malloc_device(nopt * sizeof(tfloat), *q);

  // copy data host to device
  q->memcpy(d_tx1, tx1, nopt * sizeof(tfloat));
  q->memcpy(d_tx2, tx2, nopt * sizeof(tfloat));

  q->wait();

  *x1 = d_tx1;
  *x2 = d_tx2;
  *distance_op = 0.0;

  /* Free memory */
//   _mm_free(tx1);
//   _mm_free(tx2);
}


/* Deallocate arrays */
void FreeData( queue* q, tfloat *x1, tfloat *x2 )
{
      /* Free memory */
      free(x1, q->get_context());
      free(x2, q->get_context());
}
