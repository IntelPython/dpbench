// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_l2_norm_kernel.hpp"
#include <CL/sycl.hpp>
#include <dpctl4pybind11.hpp>
#include <iostream>
#include <stdlib.h>
#include <type_traits>
#include <vector>

using namespace sycl;
namespace py = pybind11;

void l2_norm_sync(dpctl::tensor::usm_ndarray a, dpctl::tensor::usm_ndarray d)
{
    auto Queue = a.get_queue();

    auto dims = 3;
    auto npoints = a.get_size() / dims;

    if (a.get_typenum() != UAR_DOUBLE) {
        throw std::runtime_error("Expected a double precision FP array.");
    }

    l2_norm_impl(Queue, npoints, dims, a.get_data<double>(),
                 d.get_data<double>());
}

PYBIND11_MODULE(_l2_norm_sycl, m)
{
    import_dpctl();

    m.def("l2_norm", &l2_norm_sync,
          "DPC++ implementation of the l2_norm formula", py::arg("a"),
          py::arg("d"));
}
