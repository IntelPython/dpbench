// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <type_traits>

#define SIN(x) sycl::sin(x)
#define COS(x) sycl::cos(x)
#define SQRT(x) sycl::sqrt(x)
#define LOG(x) sycl::log(x)

using namespace sycl;

template <typename FpTy>
event rambo_impl(queue Queue,
                 size_t nevts,
                 size_t nout,
                 const FpTy *usmC1,
                 const FpTy *usmF1,
                 const FpTy *usmQ1,
                 FpTy *usmOutput)
{
    return Queue.submit([&](handler &h) {
        h.parallel_for<class RamboKernel>(range<1>{nevts}, [=](id<1> myID) {
            for (size_t j = 0; j < nout; j++) {
                int i = myID[0];
                size_t idx = i * nout + j;

                FpTy C = 2.0 * usmC1[idx] - 1.0;
                FpTy S = SQRT(1 - C * C);
                FpTy F = 2.0 * M_PI * usmF1[idx];
                FpTy Q = -LOG(usmQ1[idx]);

                usmOutput[idx * 4] = Q;
                usmOutput[idx * 4 + 1] = Q * S * SIN(F);
                usmOutput[idx * 4 + 2] = Q * S * COS(F);
                usmOutput[idx * 4 + 3] = Q * C;
            }
        });
    });
}
