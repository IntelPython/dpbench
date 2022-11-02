//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// The files implements a SYCL-based Python native extension for the
/// kmeans benchmark.

#include <CL/sycl.hpp>



template <typename FpTy>
void kmeans_impl(queue Queue,
		 Point* points,
		 Centroid* centroids,
		 size_t NUMBER_OF_POINTS,
		 size_t NUMBER_OF_CENTROIDS)
{
  q->submit([&](handler& h) {
      h.parallel_for<class theKernel_km>(range<1>{NUMBER_OF_CENTROIDS}, [=](id<1> myID) {
	  
	  size_t ci = myID[0];
	  centroids[ci].x = points[ci].x;
	  centroids[ci].y = points[ci].y;
	  
	});
    });
  q->wait();
  
  kmeans(q, points, centroids, NUMBER_OF_POINTS, NUMBER_OF_CENTROIDS);
}
