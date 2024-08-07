/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "constants_header.h"
#include <CL/sycl.hpp>
#include <mathimf.h>
#include <omp.h>

using namespace cl::sycl;

void l2_norm_impl(queue &Queue,
                  size_t npoints,
                  size_t dims,
                  tfloat *a,
                  tfloat *d)
{
    Queue.submit([&](handler &h) {
        h.parallel_for(range<1>{npoints}, [=](id<1> myID) {
            size_t i = myID[0];
            d[i] = 0.0;
            for (size_t k = 0; k < dims; k++) {
                d[i] += a[i * dims + k] * a[i * dims + k];
            }
            d[i] = sycl::sqrt(d[i]);
        });
    });
    Queue.wait();
}
