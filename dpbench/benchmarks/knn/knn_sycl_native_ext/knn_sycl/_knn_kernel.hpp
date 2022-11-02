//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// The files implements a SYCL-based Python native extension for the
/// knn benchmark.

#include <CL/sycl.hpp>
#include <cmath>

struct neighbors{
  double dist;
  size_t label;
};

template <typename FpTy>
sycl::event knn_impl(sycl::queue q, FpTy *d_train, size_t *d_train_labels, FpTy *d_test, size_t k, size_t classes_num, size_t train_size, size_t test_size, size_t *d_predictions, FpTy *d_votes_to_classes, size_t data_dim)
{
  sycl::event partial_hists_ev =
  q.submit([&](sycl::handler &h){
      h.parallel_for<class theKernel>(sycl::range<1>{test_size}, [=](sycl::id<1> myID) {
	  size_t i = myID[0];

	  //here k has to be 5 in order to match with numpy no. of neighbors
	  struct neighbors queue_neighbors[5];

	  //count distances
	  for (size_t j = 0; j < k; ++j) {
	    FpTy distance = 0.0;
	    for (std::size_t jj = 0; jj < data_dim; ++jj) {
	      FpTy diff = d_train[j * data_dim + jj] - d_test[i * data_dim + jj];
	      distance += diff * diff;
	    }

	    FpTy dist = sqrt(distance);

	    queue_neighbors[j].dist = dist;
	    queue_neighbors[j].label = d_train_labels[j];
	  }

	  //sort queue
	  for (size_t j = 0; j < k; ++j) {
	    // push queue
	    FpTy new_distance = queue_neighbors[j].dist;
	    FpTy new_neighbor_label = queue_neighbors[j].label;
	    size_t index = j;
	    while (index > 0 && new_distance < queue_neighbors[index-1].dist ) {
	      queue_neighbors[index] = queue_neighbors[index-1];
	      index--;

	      queue_neighbors[index].dist = new_distance;
	      queue_neighbors[index].label = new_neighbor_label;
	    }
	  }

	  for (size_t j = k; j < train_size; ++j) {
	    FpTy distance = 0.0;
	    for (std::size_t jj = 0; jj < data_dim; ++jj) {
	      FpTy diff = d_train[j * data_dim + jj] - d_test[i * data_dim + jj];
	      distance += diff * diff;
	    }

	    FpTy dist = sqrt(distance);

	    if (dist < queue_neighbors[k-1].dist) {
	      queue_neighbors[k-1].dist = dist;
	      queue_neighbors[k-1].label = d_train_labels[j];

	      //push queue
	      FpTy new_distance = queue_neighbors[k-1].dist;
	      FpTy new_neighbor_label = queue_neighbors[k-1].label;
	      size_t index = k-1;

	      while (index > 0 && new_distance < queue_neighbors[index-1].dist) {
		queue_neighbors[index] = queue_neighbors[index-1];
		index--;

		queue_neighbors[index].dist = new_distance;
		queue_neighbors[index].label = new_neighbor_label;
	      }
	    }
	  }

	  // simple vote
	  for (size_t j = 0; j < k; ++j) {
	    d_votes_to_classes[i*classes_num + queue_neighbors[j].label]++;
	  }

	  size_t max_ind = 0;
	  FpTy max_value = 0.0;

	  for (size_t j = 0; j < classes_num; ++j) {
	    if (d_votes_to_classes[i*classes_num + j] > max_value ) {
	      max_value = d_votes_to_classes[i*classes_num + j];
	      max_ind = j;
	    }
	  }
	  d_predictions[i] =  max_ind;
	});
    });
  return partial_hists_ev;
}
