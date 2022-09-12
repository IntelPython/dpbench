//==- _black_scholes_sycl.cpp - Python native extension of Black-Scholes   ===//
//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// \file
/// The files implements a SYCL-based Python native extension for the
/// black-scholes benchmark.

#include "_black_scholes_kernel.hpp"
#include <CL/sycl.hpp>
#include <dpctl4pybind11.hpp>
#include <iostream>
#include <stdlib.h>
#include <type_traits>
#include <vector>

using namespace sycl;
namespace py = pybind11;
namespace
{

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

void black_scholes_sync(size_t /**/,
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

    black_scholes_impl(Queue, nopt, price.get_data<double>(),
                       strike.get_data<double>(), t.get_data<double>(), rate,
                       volatility, call.get_data<double>(),
                       put.get_data<double>());
}

PYBIND11_MODULE(_black_scholes_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("black_scholes", &black_scholes_sync,
          "DPC++ implementation of the Black-Scholes formula", py::arg("nopt"),
          py::arg("price"), py::arg("strike"), py::arg("t"), py::arg("rate"),
          py::arg("vol"), py::arg("call"), py::arg("put"));
}
