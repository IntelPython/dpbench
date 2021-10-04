/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <omp.h>
#include "euro_opt.h"

void call_gpairs( int npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2,tfloat* rbins,tfloat* results_test) {

  int nbins = DEFAULT_NBINS;

#pragma omp parallel for simd
  for (unsigned int i = 0; i < npoints; i++) {

    tfloat px = x1[i];
    tfloat py = y1[i];
    tfloat pz = z1[i];
    tfloat pw = w1[i];
    for (unsigned int j = 0; j < npoints; j++) {
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
#pragma omp atomic update
	results_test[k-1] += wprod;
	k = k-1;
	if (k <=0) break;
      }
    }
  }
}
