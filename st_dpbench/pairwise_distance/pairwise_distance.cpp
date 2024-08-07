/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "constants_header.h"
#include <CL/sycl.hpp>
#include <mathimf.h>

using namespace cl::sycl;

#ifdef __DO_FLOAT__
#define SQRT(x) sqrtf(x)
#else
#define SQRT(x) sqrt(x)
#endif

void pairwise_distance(queue &q,
                       size_t nopt,
                       tfloat *p1,
                       tfloat *p2,
                       tfloat *distance_op,
                       size_t x1_npoints,
                       size_t x2_npoints,
                       size_t ndims)
{

    q.submit([&](handler &h) {
        h.parallel_for(range<2>{x1_npoints, x2_npoints}, [=](id<2> myID) {
            auto i = myID[0];
            auto j = myID[1];
            tfloat d = 0.0;
            for (size_t k = 0; k < ndims; k++) {
                auto tmp = p1[i * ndims + k] - p2[j * ndims + k];
                d += tmp * tmp;
            }
            distance_op[i * x2_npoints + j] = sycl::sqrt(d);
        });
    });

    q.wait();
}
