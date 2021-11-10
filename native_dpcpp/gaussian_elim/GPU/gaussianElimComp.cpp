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


void ForwardSub(queue* q, tfloat *d_a, tfloat *d_b, tfloat *d_m, int size, int *globalWorksizeFan1, int *localWorksizeFan1Buf, int *globalWorksizeFan2, int *localWorksizeFan2Buf)
{
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
  q->wait();
}
