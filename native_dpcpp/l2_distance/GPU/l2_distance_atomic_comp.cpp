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

  q->submit([&](handler& h) {
      h.parallel_for<class theKernel>(range<1>{nopt}, [=](id<1> myID) {
	  sycl::ext::oneapi::atomic_ref<tfloat, sycl::ext::oneapi::memory_order::relaxed,
					sycl::ext::oneapi::memory_scope::device,
					sycl::access::address_space::global_space>atomic_data(sum[0]);
	  size_t i = myID[0];
	  tfloat a = x1[i] - x2[i];
	  tfloat b = a * a;
	  atomic_data += b;
        });
    }).wait();

  *distance_op = SQRT(*sum);
}
