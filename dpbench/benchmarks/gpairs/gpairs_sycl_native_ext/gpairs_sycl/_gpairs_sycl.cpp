// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_gpairs_kernel.hpp"
#include <dpctl4pybind11.hpp>

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
    }
    return true;
}

void gpairs_sync(size_t nopt,
                 size_t nbins,
                 dpctl::tensor::usm_ndarray x1,
                 dpctl::tensor::usm_ndarray y1,
                 dpctl::tensor::usm_ndarray z1,
                 dpctl::tensor::usm_ndarray w1,
                 dpctl::tensor::usm_ndarray x2,
                 dpctl::tensor::usm_ndarray y2,
                 dpctl::tensor::usm_ndarray z2,
                 dpctl::tensor::usm_ndarray w2,
                 dpctl::tensor::usm_ndarray rbins,
                 dpctl::tensor::usm_ndarray results)
{
    if (!ensure_compatibility(x1, y1, z1, w1, x2, y2, z2, w2, rbins, results))
        throw std::runtime_error("Input arrays are not acceptable.");

    if (x1.get_typenum() != UAR_FLOAT && x1.get_typenum() != UAR_DOUBLE) {
        throw std::runtime_error("Expected a FP array.");
    }

    if (x1.get_typenum() == UAR_FLOAT) {
        sycl::event res_ev = gpairs_impl(
            x1.get_queue(), nopt, nbins, x1.get_data<float>(),
            y1.get_data<float>(), z1.get_data<float>(), w1.get_data<float>(),
            x2.get_data<float>(), y2.get_data<float>(), z2.get_data<float>(),
            w2.get_data<float>(), rbins.get_data<float>(),
            results.get_data<float>());
        res_ev.wait();
    }
    else {
        sycl::event res_ev = gpairs_impl(
            x1.get_queue(), nopt, nbins, x1.get_data<double>(),
            y1.get_data<double>(), z1.get_data<double>(), w1.get_data<double>(),
            x2.get_data<double>(), y2.get_data<double>(), z2.get_data<double>(),
            w2.get_data<double>(), rbins.get_data<double>(),
            results.get_data<double>());
        res_ev.wait();
    }
}

PYBIND11_MODULE(_gpairs_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("gpairs", &gpairs_sync, "DPC++ implementation of gpairs kernel",
          py::arg("nopt"), py::arg("nbins"), py::arg("x1"), py::arg("y1"),
          py::arg("z1"), py::arg("w1"), py::arg("x2"), py::arg("y2"),
          py::arg("z2"), py::arg("w2"), py::arg("rbins"), py::arg("results"));
}
