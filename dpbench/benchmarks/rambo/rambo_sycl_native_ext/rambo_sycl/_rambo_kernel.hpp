// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

// For M_PI
#define _USE_MATH_DEFINES

#include <CL/sycl.hpp>
#include <math.h>
#include <stdlib.h>
#include <type_traits>

using namespace sycl;

template <typename FpTy> class RamboKernel;

template <typename FpTy>
event rambo_impl(queue Queue,
                 size_t nevts,
                 size_t nout,
                 const FpTy *usmC1,
                 const FpTy *usmF1,
                 const FpTy *usmQ1,
                 FpTy *usmOutput)
{
    constexpr FpTy pi_v = M_PI;
    return Queue.submit([&](handler &h) {
        h.parallel_for<RamboKernel<FpTy>>(range<1>{nevts}, [=](id<1> myID) {
            for (size_t j = 0; j < nout; j++) {
                int i = myID[0];
                size_t idx = i * nout + j;

                FpTy C = 2 * usmC1[idx] - 1;
                FpTy S = sycl::sqrt(1 - C * C);
                FpTy F = 2 * pi_v * usmF1[idx];
                FpTy Q = -sycl::log(usmQ1[idx]);

                usmOutput[idx * 4] = Q;
                usmOutput[idx * 4 + 1] = Q * S * sycl::sin(F);
                usmOutput[idx * 4 + 2] = Q * S * sycl::cos(F);
                usmOutput[idx * 4 + 3] = Q * C;
            }
        });
    });
}
