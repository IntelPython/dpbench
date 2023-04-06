// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <mathimf.h>

using namespace sycl;

#ifdef __DO_FLOAT__
#define SQRT(x) sqrtf(x)
#else
#define SQRT(x) sqrt(x)
#endif

template <typename FpTy>
void pairwise_distance_impl(queue Queue,
                            size_t npoints,
                            size_t ndims,
                            const FpTy *p1,
                            const FpTy *p2,
                            FpTy *distance_op)
{
    Queue.submit([&](handler &h) {
        h.parallel_for<class PairwiseDistanceKernel>(
            range<1>{npoints}, [=](id<1> myID) {
                size_t i = myID[0];
                for (size_t j = 0; j < npoints; j++) {
                    FpTy d = 0.;
                    for (size_t k = 0; k < ndims; k++) {
                        auto tmp = p1[i * ndims + k] - p2[j * ndims + k];
                        d += tmp * tmp;
                    }
                    if (d != 0.0) {
                        distance_op[i * npoints + j] = sqrt(d);
                    }
                }
            });
    });

    Queue.wait();
}
