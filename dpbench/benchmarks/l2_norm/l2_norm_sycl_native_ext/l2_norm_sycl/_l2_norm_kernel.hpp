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
void l2_norm_impl(queue Queue,
                  size_t npoints,
                  size_t dims,
                  const FpTy *a,
                  FpTy *d)
{
    Queue
        .submit([&](handler &h) {
            h.parallel_for<class theKernel>(range<1>{npoints}, [=](id<1> myID) {
                size_t i = myID[0];
                for (size_t k = 0; k < dims; k++) {
                    d[i] += a[i * dims + k] * a[i * dims + k];
                }
                d[i] = sqrt(d[i]);
            });
        })
        .wait();
}
