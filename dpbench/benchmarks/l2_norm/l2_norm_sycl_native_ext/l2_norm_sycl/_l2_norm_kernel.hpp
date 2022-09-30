//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// The files implements a SYCL-based Python native extension for the
/// l2-norm benchmark.

#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <type_traits>

using namespace sycl;

template <typename FpTy>
void l2_norm_impl(queue Queue, size_t ndims, const FpTy *a, FpTy *d)
{
    FpTy *sum = malloc_device<FpTy>(1, Queue);

    Queue.fill<FpTy>(sum, 0., 1).wait();

    Queue
        .submit([&](handler &h) {
            auto sumr = sycl::reduction(sum, sycl::plus<>());
            h.parallel_for<class theKernel>(range<1>{ndims}, sumr,
                                            [=](id<1> myID, auto &sumr_arg) {
                                                size_t i = myID[0];
                                                sumr_arg += a[i] * a[i];
                                            });
        })
        .wait();

    Queue.copy<FpTy>(sum, d, 1).wait();
    sycl::free(sum, Queue);
    *d = sqrt(*d);
}
