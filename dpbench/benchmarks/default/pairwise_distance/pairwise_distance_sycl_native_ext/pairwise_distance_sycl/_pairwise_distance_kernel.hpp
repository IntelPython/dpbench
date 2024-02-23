// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>

using namespace sycl;

template <typename T> class PairwiseDistanceKernel;

template <typename FpTy>
void pairwise_distance_impl(queue Queue,
                            size_t x1_npoints,
                            size_t x2_npoints,
                            size_t ndims,
                            const FpTy *p1,
                            const FpTy *p2,
                            FpTy *distance_op)
{
    Queue.submit([&](handler &h) {
        h.parallel_for<PairwiseDistanceKernel<FpTy>>(
            range<2>{x1_npoints, x2_npoints}, [=](id<2> myID) {
                auto i = myID[0];
                auto j = myID[1];
                FpTy d = 0.;
                for (size_t k = 0; k < ndims; k++) {
                    auto tmp = p1[i * ndims + k] - p2[j * ndims + k];
                    d += tmp * tmp;
                }
                distance_op[i * x2_npoints + j] = sycl::sqrt(d);
            });
    });

    Queue.wait();
}
