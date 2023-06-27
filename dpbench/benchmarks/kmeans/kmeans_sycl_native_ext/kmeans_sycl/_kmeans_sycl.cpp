// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_kmeans_kernel.hpp"
#include <CL/sycl.hpp>
#include <dpctl4pybind11.hpp>

#include <iostream>
#include <stdio.h>

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
                 dpctl::tensor::usm_ndarray arrayCnumpoint,
                 size_t niters)
{
    if (!ensure_compatibility(arrayP, arrayPclusters, arrayC, arrayCnumpoint))
        throw std::runtime_error("Input arrays are not acceptable.");

    if (arrayP.get_typenum() != arrayC.get_typenum()) {
        throw std::runtime_error("All arrays must have the same precision");
    }

    if (arrayPclusters.get_typenum() != arrayCnumpoint.get_typenum()) {
        throw std::runtime_error("All arrays must have the same precision");
    }

    auto npoints = arrayP.get_shape(0);
    auto ncentroids = arrayC.get_shape(0);
    auto ndims = arrayC.get_shape(1);

#define call_kmeans(dims, ftyp, ityp)                                          \
    kmeans_impl<ftyp, ityp, dims>(                                             \
        arrayP.get_queue(), arrayP.get_data<ftyp>(),                           \
        arrayPclusters.get_data<ityp>(), arrayC.get_data<ftyp>(),              \
        arrayCnumpoint.get_data<ityp>(), niters, npoints, ncentroids, 0);

#define dispatch_kmeans_ftype_itype(ftyp, ityp)                                \
    {                                                                          \
        if (ndims == 2) {                                                      \
            call_kmeans(2, ftyp, ityp);                                        \
        }                                                                      \
        else if (ndims == 3) {                                                 \
            call_kmeans(3, ftyp, ityp);                                        \
        }                                                                      \
        else if (ndims == 4) {                                                 \
            call_kmeans(4, ftyp, ityp);                                        \
        }                                                                      \
        else {                                                                 \
            throw std::runtime_error("Unsupported ndims");                     \
        }                                                                      \
    }

#define dispatch_kmeans(ityp)                                                  \
    {                                                                          \
        if (arrayP.get_typenum() == UAR_DOUBLE) {                              \
            dispatch_kmeans_ftype_itype(double, ityp);                         \
        }                                                                      \
        else if (arrayP.get_typenum() == UAR_FLOAT) {                          \
            dispatch_kmeans_ftype_itype(float, ityp);                          \
        }                                                                      \
        else {                                                                 \
            throw std::runtime_error("Unsupported type");                      \
        }                                                                      \
    }

    if (arrayCnumpoint.get_elemsize() == 4 and
        (arrayCnumpoint.get_typenum() == UAR_INT or
         arrayCnumpoint.get_typenum() == UAR_LONG))
    {
        dispatch_kmeans(int32_t);
    }
    else if (arrayCnumpoint.get_elemsize() == 8 and
             (arrayCnumpoint.get_typenum() == UAR_LONG or
              arrayCnumpoint.get_typenum() == UAR_LONGLONG))
    {
        dispatch_kmeans(int64_t);
    }
    else {
        throw std::runtime_error("Unsupported type");
    }

#undef dispatch_kmeans
#undef call_kmeans
}

PYBIND11_MODULE(_kmeans_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("kmeans", &kmeans_sync, "DPC++ implementation of Kmeans",
          py::arg("arrayP"), py::arg("arrayPclusters"), py::arg("arrayC"),
          py::arg("arrayCnumpoint"), py::arg("niters"));
}
