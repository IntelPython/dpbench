//==- _kmeans_sycl.cpp - Python native extension of Kmeans   ===//
//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// \file
/// The files implements a SYCL-based Python native extension for the
/// kmeans benchmark.

#include "_kmeans_kernel.hpp"
#include <CL/sycl.hpp>
#include <dpctl4pybind11.hpp>

template <typename... Args> bool ensure_compatibility(const Args &...args)
{
    std::vector<dpctl::tensor::usm_ndarray> arrays = {args...};

    auto arr = arrays.at(0);

    for (auto &arr : arrays) {
        if (!(arr.get_flags() & (USM_ARRAY_C_CONTIGUOUS))) {
            std::cerr << "All arrays need to be C contiguous.\n";
            return false;
        }
    }
    return true;
}

void kmeans_sync(dpctl::tensor::usm_ndarray arrayP,
                 dpctl::tensor::usm_ndarray arrayPclusters,
                 dpctl::tensor::usm_ndarray arrayC,
                 dpctl::tensor::usm_ndarray arrayCsum,
                 dpctl::tensor::usm_ndarray arrayCnumpoint,
                 size_t niters,
                 size_t npoints,
                 size_t ndims,
                 size_t ncentroids)
{
    if (!ensure_compatibility(arrayP, arrayPclusters, arrayC, arrayCsum,
                              arrayCnumpoint))
        throw std::runtime_error("Input arrays are not acceptable.");

    if (arrayP.get_typenum() != UAR_DOUBLE ||
        arrayC.get_typenum() != UAR_DOUBLE ||
        arrayCsum.get_typenum() != UAR_DOUBLE)
    {
        throw std::runtime_error("Expected a double precision FP array.");
    }

    kmeans_impl(arrayP.get_queue(), arrayP.get_data<double>(),
                arrayPclusters.get_data<size_t>(), arrayC.get_data<double>(),
                arrayCsum.get_data<double>(), arrayCnumpoint.get_data<size_t>(),
                niters, npoints, ndims, ncentroids);
}

PYBIND11_MODULE(_kmeans_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("kmeans", &kmeans_sync, "DPC++ implementation of Kmeans",
          py::arg("arrayP"), py::arg("arrayPclusters"), py::arg("arrayC"),
          py::arg("arrayCsum"), py::arg("arrayCnumpoint"), py::arg("niters"),
          py::arg("npoints"), py::arg("ndims"), py::arg("ncentroids"));
}
