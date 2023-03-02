// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_dbscan_kernel.hpp"
#include <CL/sycl.hpp>
#include <dpctl4pybind11.hpp>
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
    }
    return true;
}

} // namespace

size_t dbscan_sync(size_t n_samples,
                   size_t n_features,
                   dpctl::tensor::usm_ndarray data,
                   double eps,
                   size_t min_pts,
                   dpctl::tensor::usm_ndarray assignments)
{
    auto queue = data.get_queue();

    if (!ensure_compatibility(data, assignments))
        throw std::runtime_error("Input arrays are not acceptable.");

    return dbscan_impl<double>(queue, n_samples, n_features,
                               data.get_data<double>(), eps, min_pts,
                               assignments.get_data<size_t>());
}

PYBIND11_MODULE(_dbscan_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("dbscan", &dbscan_sync, "DPC++ implementation of DBSCAN",
          py::arg("n_samples"), py::arg("n_features"), py::arg("data"),
          py::arg("eps"), py::arg("min_pts"), py::arg("assignments"));
}
