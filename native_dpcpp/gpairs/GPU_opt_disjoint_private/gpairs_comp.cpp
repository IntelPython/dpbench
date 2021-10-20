/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <CL/sycl.hpp>
#include "euro_opt.h"

#define LWS 64

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

void call_gpairs_opt( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2,tfloat* rbins,tfloat* results_test) {

  int nbins = DEFAULT_NBINS;

  q->submit([&](handler& h) {
      
      h.parallel_for<class ComputeKernel>(sycl::nd_range(sycl::range{npoints}, sycl::range{LWS}), [=](sycl::nd_item<1> myID) {
	  //h.parallel_for<class ComputeKernel>(range<1>{npoints}, [=](id<1> myID) {
  	  size_t i = myID.get_global_id(0);
	  //size_t i = myID[0];

	  tfloat results_pvt[DEFAULT_NBINS-1] = {0};
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

  	    int k = nbins - 1;
  	    while(dsq <= rbins_pvt[k]) {
  	      //results_test[i*(nbins-1)+(k-1)] += wprod;
	      results_pvt[k-1] += wprod;
  	      k = k-1;
  	      if (k <=0) break;
  	    }
  	  }
	  for (int j=0; j < nbins-1;j++) results_test[i*(nbins-1)+j] = results_pvt[j];
  	});
    });

  q->wait();

  q->submit([&](handler& h) {
      h.parallel_for<class MergeKernel>(nbins, [=](id<1> myID) {
  	  int col_id = myID[0];

  	  for (size_t i = 1; i < npoints; i++) {
  	    results_test[col_id] += results_test[i*(nbins-1)+col_id];
  	  }

  	});
    });

  q->wait();
	
}

// void call_gpairs_naieve_opt( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2,tfloat* rbins,tfloat* results_test) {
//   int nbins = DEFAULT_NBINS;
//   for (size_t i = 0; i < npoints; i++) {
//     tfloat px = x1[i];
//     tfloat py = y1[i];
//     tfloat pz = z1[i];
//     tfloat pw = w1[i];
//     for (size_t j = 0; j < npoints; j++) {
//       tfloat qx = x2[j];
//       tfloat qy = y2[j];
//       tfloat qz = z2[j];
//       tfloat qw = w2[j];
//       tfloat dx = px - qx;
//       tfloat dy = py - qy;
//       tfloat dz = pz - qz;
//       tfloat wprod = pw * qw;
//       tfloat dsq = dx*dx + dy*dy + dz*dz;

//       int k = nbins - 1;
//       while(dsq <= rbins[k]) {
// 	results_test[i*(nbins-1)+(k-1)] += wprod;
// 	k = k-1;
// 	if (k <=0) break;
//       }
//     }
//   }

//   for (int col_id = 0; col_id < nbins; col_id++) {    
//     for (size_t i = 1; i < npoints; i++) {
//       results_test[col_id] += results_test[i*(nbins-1)+col_id];
//     }
//   }    
// }
