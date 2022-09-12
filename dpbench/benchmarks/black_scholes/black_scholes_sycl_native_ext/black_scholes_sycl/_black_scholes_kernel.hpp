//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// The files implements a SYCL-based Python native extension for the
/// black-scholes benchmark.

#include <stdlib.h>
#include <sycl.hpp>
#include <type_traits>

#ifdef __DO_FLOAT__
#define EXP(x) expf(x)
#define LOG(x) logf(x)
#define SQRT(x) sqrtf(x)
#define ERF(x) erff(x)
#define INVSQRT(x) 1.0f / sqrtf(x)

#define QUARTER 0.25f
#define HALF 0.5f
#define TWO 2.0f
#else
#define EXP(x) sycl::exp(x)
#define LOG(x) sycl::log(x)
#define SQRT(x) sycl::sqrt(x)
#define ERF(x) sycl::erf(x)
#define INVSQRT(x) 1.0 / sycl::sqrt(x)

#define QUARTER 0.25
#define HALF 0.5
#define TWO 2.0
#endif

using namespace sycl;

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
    auto e = Queue.submit([&](handler &h) {
        h.parallel_for<class BlackScholesKernel>(
            range<1>{nopt}, [=](id<1> myID) {
                FpTy mr = -rate;
                FpTy sig_sig_two = volatility * volatility * TWO;
                int i = myID[0];
                FpTy a, b, c, y, z, e;
                FpTy d1, d2, w1, w2;

                a = LOG(price[i] / strike[i]);
                b = t[i] * mr;
                z = t[i] * sig_sig_two;
                c = QUARTER * z;
                y = INVSQRT(z);
                w1 = (a - b + c) * y;
                w2 = (a - b - c) * y;
                d1 = ERF(w1);
                d2 = ERF(w2);
                d1 = HALF + HALF * d1;
                d2 = HALF + HALF * d2;
                e = EXP(b);
                call[i] = price[i] * d1 - strike[i] * e * d2;
                put[i] = call[i] - price[i] + strike[i] * e;
            });
    });
    e.wait();
}
