//==- _rambo_sycl.cpp - Python native extension of Rambo   ===//
//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// \file
/// The files implements a SYCL-based Python native extension for the
/// Rambo benchmark.

#include "_rambo_kernel.hpp"
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

void rambo_sync(size_t nevts, size_t nout, dpctl::tensor::usm_ndarray output)
{
    auto Queue = output.get_queue();

    auto typenum = output.get_typenum();

    if (typenum != UAR_DOUBLE) {
        throw std::runtime_error("Expected a double precision FP array.");
    }

    const size_t inputSize = nevts * nout;
    std::vector<double> C1(inputSize), F1(inputSize), Q1(inputSize);

    e2.seed(777);
    for (auto i = 0; i < nevts; i++) {
        for (auto j = 0; j < nout; j++) {
            C1[i * nout + j] = genRand<double>();
            F1[i * nout + j] = genRand<double>();
            Q1[i * nout + j] = genRand<double>() * genRand<double>();
        }
    }

    double *usmC1 = malloc_device<double>(inputSize, Queue);
    double *usmF1 = malloc_device<double>(inputSize, Queue);
    double *usmQ1 = malloc_device<double>(inputSize, Queue);

    Queue.copy<double>(&C1[0], usmC1, inputSize).wait();
    Queue.copy<double>(&F1[0], usmF1, inputSize).wait();
    Queue.copy<double>(&Q1[0], usmQ1, inputSize).wait();

    rambo_impl(Queue, nevts, nout, usmC1, usmF1, usmQ1,
               output.get_data<double>());

    free(usmC1, Queue);
    free(usmF1, Queue);
    free(usmQ1, Queue);
}

PYBIND11_MODULE(_rambo_sycl, m)
{
    import_dpctl();

    m.def("rambo", &rambo_sync, "DPC++ implementation of the Rambo formula",
          py::arg("nevts"), py::arg("nout"), py::arg("output"));
}
