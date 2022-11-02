//==- _knn_sycl.cpp - Python native extension of Black-Scholes   ===//
//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// \file
/// The files implements a SYCL-based Python native extension for the
/// knn benchmark.

#include <dpctl4pybind11.hpp>
#include "_knn_kernel.hpp"

#include <iostream>

void knn_sync(dpctl::tensor::usm_ndarray x_train,
	      dpctl::tensor::usm_ndarray y_train,
	      dpctl::tensor::usm_ndarray x_test,
	      size_t k, size_t classes_num, size_t train_size, size_t test_size,
	      dpctl::tensor::usm_ndarray predictions,
	      dpctl::tensor::usm_ndarray votes_to_classes, size_t data_dim)
{
  sycl::event res_ev = knn_impl(x_train.get_queue(), x_train.get_data<double>(),
				y_train.get_data<size_t>(), x_test.get_data<double>(),
				k, classes_num, train_size, test_size,
				predictions.get_data<size_t>(), votes_to_classes.get_data<double>(), data_dim);
  res_ev.wait();

  // size_t *d_predictions = predictions.get_data<size_t>();
  // size_t *h_predictions = new size_t[test_size];
  // sycl::queue q = x_train.get_queue();
  // q.memcpy(h_predictions, d_predictions, test_size * sizeof(size_t));
  // q.wait();

  // for (size_t i = 0; i < test_size; i++) {
  //   std::cout << h_predictions[i] << std::endl;
  // }
}

PYBIND11_MODULE(_knn_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("knn", &knn_sync,
          "DPC++ implementation of knn kernel", py::arg("x_train"),
          py::arg("y_train"), py::arg("x_test"), py::arg("k"), py::arg("classes_num"),
	  py::arg("train_size"), py::arg("test_size"),
          py::arg("predictions"), py::arg("votes_to_classes"), py::arg("data_dim"));
}
