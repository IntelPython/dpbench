//==- _knn_sycl.cpp - Python native extension of Black-Scholes   ===//
//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// \file
/// The files implements a SYCL-based Python native extension for the
/// knn benchmark.

#include "_knn_kernel.hpp"
#include <dpctl4pybind11.hpp>

#include <iostream>

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

void knn_sync(dpctl::tensor::usm_ndarray x_train,
              dpctl::tensor::usm_ndarray y_train,
              dpctl::tensor::usm_ndarray x_test,
              size_t k,
              size_t classes_num,
              size_t train_size,
              size_t test_size,
              dpctl::tensor::usm_ndarray predictions,
              dpctl::tensor::usm_ndarray votes_to_classes,
              size_t data_dim)
{

    if (!ensure_compatibility(x_train, y_train, x_test, predictions,
                              votes_to_classes))
        throw std::runtime_error("Input arrays are not acceptable.");

    if (x_train.get_typenum() != UAR_DOUBLE) {
        throw std::runtime_error("Expected a double precision FP array.");
    }

    sycl::event res_ev = knn_impl(
        x_train.get_queue(), x_train.get_data<double>(),
        y_train.get_data<size_t>(), x_test.get_data<double>(), k, classes_num,
        train_size, test_size, predictions.get_data<size_t>(),
        votes_to_classes.get_data<double>(), data_dim);
    res_ev.wait();
}

PYBIND11_MODULE(_knn_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("knn", &knn_sync, "DPC++ implementation of knn kernel",
          py::arg("x_train"), py::arg("y_train"), py::arg("x_test"),
          py::arg("k"), py::arg("classes_num"), py::arg("train_size"),
          py::arg("test_size"), py::arg("predictions"),
          py::arg("votes_to_classes"), py::arg("data_dim"));
}
