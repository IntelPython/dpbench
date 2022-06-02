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
      h.parallel_for<class theKernel>(range<1>{nopt}, [=](id<1> myID) {
	  sycl::atomic_ref<tfloat, sycl::memory_order::relaxed,
  	      			        sycl::memory_scope::device,
					sycl::access::address_space::global_space>atomic_data(d_sum[0]);

	  size_t i = myID[0];
	  tfloat a = d_x1[i] - d_x2[i];
	  tfloat b = a * a;
	  atomic_data += b;
	});
    }).wait();

  q->memcpy(&sum, d_sum, sizeof(tfloat));
  q->wait();

  *distance_op = SQRT(sum);

  free(d_x1, q->get_context());
  free(d_x2, q->get_context());
  free(d_sum, q->get_context());
}
