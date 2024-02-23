// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <type_traits>

using namespace sycl;

template <typename FpTy> class BlackScholesKernel;

template <typename FpTy>
void black_scholes_impl(queue Queue,
                        size_t nopt,
                        const FpTy *price,
                        const FpTy *strike,
                        const FpTy *t,
                        FpTy rate,
                        FpTy volatility,
                        FpTy *call,
                        FpTy *put)
{
    constexpr FpTy _0_25 = 0.25;
    constexpr FpTy _0_5 = 0.5;

    auto e = Queue.submit([&](handler &h) {
        h.parallel_for<BlackScholesKernel<FpTy>>(
            range<1>{nopt}, [=](id<1> myID) {
                FpTy mr = -rate;
                FpTy sig_sig_two = volatility * volatility * 2;
                int i = myID[0];
                FpTy a, b, c, y, z, e;
                FpTy d1, d2, w1, w2;

                a = sycl::log(price[i] / strike[i]);
                b = t[i] * mr;
                z = t[i] * sig_sig_two;
                c = _0_25 * z;
                y = sycl::rsqrt(z);
                w1 = (a - b + c) * y;
                w2 = (a - b - c) * y;
                d1 = sycl::erf(w1);
                d2 = sycl::erf(w2);
                d1 = _0_5 + _0_5 * d1;
                d2 = _0_5 + _0_5 * d2;
                e = sycl::exp(b);
                call[i] = price[i] * d1 - strike[i] * e * d2;
                put[i] = call[i] - price[i] + strike[i] * e;
            });
    });
    e.wait();
}
