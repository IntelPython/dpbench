/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

  #include <CL/sycl.hpp>
  #include <omp.h>
  #include <mathimf.h>
  #include "constants_header.h"

  using namespace cl::sycl;

  #ifdef __DO_FLOAT__
  #   define SQRT(x)     sqrtf(x)
  #else
  #   define SQRT(x)     sqrt(x)
  #endif

  void l2_distance( queue* q, size_t nopt, tfloat* x1, tfloat* x2, tfloat* distance_op) {
    tfloat sum = 0.0;

    tfloat* d_x1, *d_x2, *d_sum;
    d_x1 = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    d_x2 = (tfloat*)malloc_shared( nopt * sizeof(tfloat), *q);
    d_sum = (tfloat*)malloc_shared( sizeof(tfloat), *q);

    q->memcpy(d_x1, x1, nopt * sizeof(tfloat));
    q->memcpy(d_x2, x2, nopt * sizeof(tfloat));
    q->memcpy(d_sum, &sum, sizeof(tfloat));

    q->wait();

    q->submit([&](handler& h) {
        auto sumr = sycl::ONEAPI::reduction(d_sum, sycl::ONEAPI::plus<>());

        h.parallel_for<class theKernel>(sycl::nd_range<1>{nopt, 256}, sumr, [=](sycl::nd_item<1> item, auto& sumr_arg) {
            size_t i = item.get_global_id(0);
            tfloat a = d_x1[i] - d_x2[i];
            tfloat b = a * a;
            sumr_arg += b;
          });
      }).wait();

    q->memcpy(&sum, d_sum, sizeof(tfloat));
    q->wait();

    *distance_op = SQRT(sum);

    free(d_x1, q->get_context());
    free(d_x2, q->get_context());
    free(d_sum, q->get_context());
  }
