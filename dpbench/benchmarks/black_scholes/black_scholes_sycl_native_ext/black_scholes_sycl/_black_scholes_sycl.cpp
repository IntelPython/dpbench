//==- _black_scholes_sycl.cpp - Python native extension of Black-Scholes   ===//
//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : BSD - 3 - Clause
///
/// \file
/// The files implements a SYCL-based Python native extension for the
/// black-scholes benchmark.

#include <CL/sycl.hpp>
#include <dpctl4pybind11.hpp>
#include <stdlib.h>
#include <type_traits>
#include <vector>

using namespace sycl;
namespace py = pybind11;

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

namespace
{

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
    // timer ON
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
    // timer OFF
}

template <typename... Args> bool ensure_compatibility(const Args &...args)
{
    std::vector<dpctl::tensor::usm_ndarray> arrays = {args...};

    auto arr = arrays.at(0);
    auto q = arr.get_queue();
    auto type_flag = arr.get_typenum();
    auto arr_size = arr.get_size();

    for (auto &arr : arrays) {
        if (!(arr.get_flags() & (USM_ARRAY_C_CONTIGUOUS))) {
            std::cerr << "All arrays need to be C contiguous.\n";
            return false;
        }
        if (q != arr.get_queue()) {
            std::cerr << "All arrays should be in same SYCL queue.\n";
            return false;
        }
        if (arr.get_typenum() != type_flag) {
            std::cerr << "All arrays should be of same elemental type.\n";
            return false;
        }
        if (arr.get_ndim() > 1) {
            std::cerr << "All arrays expected to be single-dimensional.\n";
            return false;
        }
        if (arr.get_size() != arr_size) {
            std::cerr << "All arrays expected to be of same size.\n";
            return false;
        }
    }
    return true;
}

} // namespace

void
black_scholes_sync(size_t /**/,
                   dpctl::tensor::usm_ndarray price,
                   dpctl::tensor::usm_ndarray strike,
                   dpctl::tensor::usm_ndarray t,
                   double rate,
                   double volatility,
                   dpctl::tensor::usm_ndarray call,
                   dpctl::tensor::usm_ndarray put)
{
    sycl::event res_ev;
    auto Queue = price.get_queue();
    auto nopt = price.get_size();
    auto typenum = price.get_typenum();

    if (!ensure_compatibility(price, strike, t, call, put))
        throw std::runtime_error("Input arrays are not acceptable.");

    if (typenum != UAR_DOUBLE) {
        throw std::runtime_error("Expected a double precision FP array.");
    }

    black_scholes_impl(
        Queue, nopt, (double *)price.get_data(), (double *)strike.get_data(),
        (double *)t.get_data(), rate, volatility, (double *)call.get_data(),
        (double *)put.get_data());
}

PYBIND11_MODULE(_black_scholes_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("black_scholes", &black_scholes_sync,
          "DPC++ implementation of the Black-Scholes formula",
          py::arg("nopt"),
          py::arg("price"),
          py::arg("strike"),
          py::arg("t"),
          py::arg("rate"),
          py::arg("vol"),
          py::arg("call"),
          py::arg("put")
    );
}
