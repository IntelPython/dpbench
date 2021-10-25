/*------------------------------------------------------
 ** Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
 /*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <CL/sycl.hpp>
#include <omp.h>
#include <mathimf.h>
#include "gaussianElim.h"

using namespace cl::sycl;


void ForwardSub(queue* q, tfloat *a, tfloat *b, tfloat *m, int size, int *globalWorksizeFan1, int *localWorksizeFan1Buf, int *globalWorksizeFan2, int *localWorksizeFan2Buf)
{
  // 1. set up kernels

  // 2. set up memory on device and send ipts data to device

  tfloat *d_a, *d_b, *d_m;
  d_a = (tfloat*)malloc_shared(size * size * sizeof(tfloat), *q);
  d_b = (tfloat*)malloc_shared(size * sizeof(tfloat), *q);
  d_m = (tfloat*)malloc_shared(size * size * sizeof(tfloat), *q);

  q->memcpy(d_a, a, size * size * sizeof(tfloat));
  q->memcpy(d_b, b, size * sizeof(tfloat));
  q->memcpy(d_m, m, size * size * sizeof(tfloat));

  q->wait();

  // 4. Setup and Run kernels
  for (int t = 0; t < (size-1); t++)
  {
    q->submit([&](handler& h)
    {
        h.parallel_for<class fan1>(
          nd_range<1>(range<1>(globalWorksizeFan1[0]),
                      range<1>(localWorksizeFan1Buf[0])), [=] (nd_item<1> item) {
          int globalId = item.get_global_id(0);
          if (globalId < size-1-t) {
            d_m[size * (globalId + t + 1) + t] =
            d_a[size * (globalId + t + 1) + t] / d_a[size * t + t];
            }
          });
        });

    q->submit([&](handler& h)
    {

        h.parallel_for<class fan2>(
          nd_range<2>(range<2>(globalWorksizeFan2[0],globalWorksizeFan2[1]),
                      range<2>(localWorksizeFan2Buf[0], localWorksizeFan2Buf[1])), [=] (nd_item<2> item) {
          int globalIdx = item.get_global_id(0);
          int globalIdy = item.get_global_id(1);

          if (globalIdx < size-1-t && globalIdy < size-t) {
            d_a[size*(globalIdx+1+t)+(globalIdy+t)] -=
            d_m[size*(globalIdx+1+t)+t] * d_a[size*t+(globalIdy+t)];

            if(globalIdy == 0){
              d_b[globalIdx+1+t] -=
              d_m[size*(globalIdx+1+t)+(globalIdy+t)] * d_b[t];
            }
          }
          });
    });
  }

  q->memcpy(a, d_a, size * size * sizeof(tfloat));
  q->memcpy(b, d_b, size * sizeof(tfloat));
  q->wait();

  free(d_a, q->get_context());
  free(d_b, q->get_context());
  free(d_m, q->get_context());
}
