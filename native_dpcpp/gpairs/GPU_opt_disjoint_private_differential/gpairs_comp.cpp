/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <CL/sycl.hpp>
#include "euro_opt.h"

#define LWS 64

using namespace cl::sycl;

void call_gpairs_opt( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2,tfloat* rbins,tfloat* results_test) {

  int nbins = DEFAULT_NBINS;

  q->submit([&](handler& h) {

      h.parallel_for<class ComputeKernel>(sycl::nd_range(sycl::range{npoints}, sycl::range{LWS}), [=](sycl::nd_item<1> myID) {
	  //h.parallel_for<class ComputeKernel>(range<1>{npoints}, [=](id<1> myID) {
	  //size_t i = myID[0];
  	  size_t i = myID.get_global_id(0);

	  tfloat results_pvt[DEFAULT_NBINS] = {0};
	  tfloat rbins_pvt[DEFAULT_NBINS];
	  for (int j=0; j < DEFAULT_NBINS; j++) rbins_pvt[j]=rbins[j];

  	  tfloat px = x1[i];
  	  tfloat py = y1[i];
  	  tfloat pz = z1[i];
  	  tfloat pw = w1[i];
  	  for (size_t j = 0; j < npoints; j++) {
  	    tfloat qx = x2[j];
  	    tfloat qy = y2[j];
  	    tfloat qz = z2[j];
  	    tfloat qw = w2[j];
  	    tfloat dx = px - qx;
  	    tfloat dy = py - qy;
  	    tfloat dz = pz - qz;
  	    tfloat wprod = pw * qw;
  	    tfloat dsq = dx*dx + dy*dy + dz*dz;

	    if (dsq <= rbins_pvt[nbins-1]) {
	      for (int k = nbins-1; k >= 0; k--) {
		if (dsq > rbins_pvt[k]) {
		  results_pvt[k+1] += wprod;
		  break;
		} else if (k == 0) results_pvt[k] += wprod;
	      }
	    }
  	  }

	  //Iterate through work-item private result from n-2->0(where n=nbins-1).
	  //For each j'th bin add it's contents to all bins from j+1 to n-1
	  for (int j = nbins-2; j >= 0; j--) {
	    for (int k = j+1; k < nbins; k++) {
	      results_pvt[k] += results_pvt[j];
	    }
	  }

	  //Propagate results from private memory to global memory
	  for (int j=0; j < nbins;j++) results_test[i*nbins+j] = results_pvt[j];
  	});
    });

  q->wait();

  q->submit([&](handler& h) {
      h.parallel_for<class MergeKernel>(nbins, [=](id<1> myID) {
  	  int col_id = myID[0];

  	  for (size_t i = 1; i < npoints; i++) {
  	    results_test[col_id] += results_test[i*nbins+col_id];
  	  }

  	});
    });

  q->wait();
}
