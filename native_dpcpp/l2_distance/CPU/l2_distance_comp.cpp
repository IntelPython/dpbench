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

void l2_distance( queue* q, size_t nopt, tfloat * x1, tfloat * x2, tfloat* distance_op ) {
  size_t i;
  tfloat *sum = (tfloat*)malloc_shared(sizeof(tfloat), *q);
  *sum = 0;

  q->submit([&](handler& h) {
      auto sumr = sycl::ONEAPI::reduction(sum, sycl::ONEAPI::plus<>());
      
      h.parallel_for<class theKernel>(sycl::nd_range<1>{nopt,256}, sumr, [=](sycl::nd_item<1> item, auto& sumr_arg) {
	  size_t i = item.get_global_id(0);
	  tfloat a = x1[i] - x2[i];
	  tfloat b = a * a;
	  sumr_arg += b;
	});
    }).wait();

  *distance_op = SQRT(*sum);      
}
