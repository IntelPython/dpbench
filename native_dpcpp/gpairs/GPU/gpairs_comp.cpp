/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <CL/sycl.hpp>
#include "euro_opt.h"

using namespace cl::sycl;

void call_gpairs_naieve( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2,tfloat* rbins,tfloat* results_test) {
  int nbins = DEFAULT_NBINS;
  for (size_t i = 0; i < npoints; i++) {
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

      int k = nbins - 1;
      while(dsq <= rbins[k]) {	  
	results_test[k-1] += wprod;
	k = k-1;
	if (k <=0) break;
      }
    }
  }
}

void call_gpairs( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2,tfloat* rbins,tfloat* results_test) {

  int nbins = DEFAULT_NBINS;
  
  q->submit([&](handler& h) {
      h.parallel_for<class theKernel>(range<1>{npoints}, [=](id<1> myID) {
  	  size_t i = myID[0];
	  
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

  	    int k = nbins - 1;
  	    while(dsq <= rbins[k]) {	  
  	      sycl::ext::oneapi::atomic_ref<tfloat, sycl::ext::oneapi::memory_order::relaxed,
  	      			       sycl::ext::oneapi::memory_scope::device,
  	      			       sycl::access::address_space::global_space>atomic_data(results_test[k-1]);
	  
  	      atomic_data += wprod;
  	      k = k-1;
  	      if (k <=0) break;
  	    }
  	  }	  
  	});
    });

  q->wait();
}
