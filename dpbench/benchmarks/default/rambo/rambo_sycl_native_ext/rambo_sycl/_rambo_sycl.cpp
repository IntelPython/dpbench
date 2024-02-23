// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

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
        if (arr.get_size() != arr_size) {
            std::cerr << "All arrays expected to be of same size.\n";
            return false;
        }
    }
    return true;
}

} // namespace

void rambo_sync(size_t nevts,
                size_t nout,
                dpctl::tensor::usm_ndarray C1,
                dpctl::tensor::usm_ndarray F1,
                dpctl::tensor::usm_ndarray Q1,
                dpctl::tensor::usm_ndarray output)
{
    auto Queue = output.get_queue();

    if (!ensure_compatibility(C1, F1, Q1))
        throw std::runtime_error("Input arrays are not acceptable.");

    if (output.get_typenum() != C1.get_typenum())
        throw std::runtime_error("Input arrays are not acceptable.");

    auto typenum = C1.get_typenum();

    if (typenum == UAR_FLOAT) {
        auto e = rambo_impl(Queue, nevts, nout, C1.get_data<float>(),
                            F1.get_data<float>(), Q1.get_data<float>(),
                            output.get_data<float>());
        e.wait();
    }
    else if (typenum == UAR_DOUBLE) {
        auto e = rambo_impl(Queue, nevts, nout, C1.get_data<double>(),
                            F1.get_data<double>(), Q1.get_data<double>(),
                            output.get_data<double>());
        e.wait();
    }
    else {
        throw std::runtime_error(
            "Expected a double or single precision FP array.");
    }
}

PYBIND11_MODULE(_rambo_sycl, m)
{
    import_dpctl();

    m.def("rambo", &rambo_sync, "DPC++ implementation of the Rambo formula",
          py::arg("nevts"), py::arg("nout"), py::arg("C1"), py::arg("F1"),
          py::arg("Q1"), py::arg("output"));
}
