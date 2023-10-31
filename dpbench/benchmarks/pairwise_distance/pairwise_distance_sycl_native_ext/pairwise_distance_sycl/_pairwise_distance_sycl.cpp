// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_pairwise_distance_kernel.hpp"
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
    }
    return true;
}

} // namespace

void pairwise_distance_sync(dpctl::tensor::usm_ndarray X1,
                            dpctl::tensor::usm_ndarray X2,
                            dpctl::tensor::usm_ndarray D)
{
    sycl::event res_ev;
    auto Queue = X1.get_queue();
    auto ndims = X1.get_shape(1);
    auto x1_npoints = X1.get_shape(0);
    auto x2_npoints = X2.get_shape(0);

    if (!ensure_compatibility(X1, X2, D))
        throw std::runtime_error("Input arrays are not acceptable.");

    if (X1.get_typenum() == UAR_FLOAT) {
        pairwise_distance_impl(Queue, x1_npoints, x2_npoints, ndims,
                               X1.get_data<float>(), X2.get_data<float>(),
                               D.get_data<float>());
    }
    else if (X1.get_typenum() == UAR_DOUBLE) {
        pairwise_distance_impl(Queue, x1_npoints, x2_npoints, ndims,
                               X1.get_data<double>(), X2.get_data<double>(),
                               D.get_data<double>());
    }
    else {
        throw std::runtime_error(
            "Expected a double or single precision FP array.");
    }
}

PYBIND11_MODULE(_pairwise_distance_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("pairwise_distance", &pairwise_distance_sync,
          "DPC++ implementation of the pairwise_distance formula",
          py::arg("X1"), py::arg("X2"), py::arg("D"));
}
