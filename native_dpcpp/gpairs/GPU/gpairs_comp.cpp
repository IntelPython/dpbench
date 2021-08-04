/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "euro_opt.h"

void call_gpairs( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2,tfloat* rbins,tfloat* results_test) {

  int nbins = DEFAULT_NBINS;
  // tfloat *d_x1, *d_y1, *d_z1, *d_w1, *d_x2, *d_y2, *d_z2, *d_w2, *d_rbins, *d_results_test;

  // d_x1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_y1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_z1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_w1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_x2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_y2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_z2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_w2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  // d_rbins = (tfloat*)malloc_device( DEFAULT_NBINS * sizeof(tfloat), *q);
  // d_results_test = (tfloat*)malloc_device( (DEFAULT_NBINS-1) * sizeof(tfloat), *q);

  // // copy data host to device
  // q->memcpy(d_x1, x1, npoints * sizeof(tfloat));
  // q->memcpy(d_y1, y1, npoints * sizeof(tfloat));
  // q->memcpy(d_z1, z1, npoints * sizeof(tfloat));
  // q->memcpy(d_w1, w1, npoints * sizeof(tfloat));
  // q->memcpy(d_x2, x2, npoints * sizeof(tfloat));
  // q->memcpy(d_y2, y2, npoints * sizeof(tfloat));
  // q->memcpy(d_z2, z2, npoints * sizeof(tfloat));
  // q->memcpy(d_w2, w2, npoints * sizeof(tfloat));
  // q->memcpy(d_rbins, rbins, DEFAULT_NBINS * sizeof(tfloat));
  // q->memcpy(d_results_test, results_test, (DEFAULT_NBINS-1) * sizeof(tfloat));
  
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
	      sycl::ONEAPI::atomic_ref<tfloat, sycl::ONEAPI::memory_order::relaxed,
				       sycl::ONEAPI::memory_scope::device,
				       sycl::access::address_space::global_space>atomic_data(results_test[k-1]);
	  
	      atomic_data += wprod;
	      k = k-1;
	      if (k <=0) break;
	    }
	  }	  
	});
    }).wait();

  // q->memcpy(results_test, d_results_test, (DEFAULT_NBINS-1) * sizeof(tfloat));

  // q->wait();
  
  // free(d_x1,q->get_context());
  // free(d_y1,q->get_context());
  // free(d_z1,q->get_context());
  // free(d_w1,q->get_context());
  // free(d_x2,q->get_context());
  // free(d_y2,q->get_context());
  // free(d_z2,q->get_context());
  // free(d_w2,q->get_context());
  // free(d_rbins,q->get_context());
}
