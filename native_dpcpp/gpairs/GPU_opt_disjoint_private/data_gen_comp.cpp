/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#define _XOPEN_SOURCE
#define _DEFAULT_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <ia32intrin.h>
#include <cmath>

#include "euro_opt.h"

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
    double r;
    drand48_r(seed, &r);
    return r*(b-a) + a;
}

template <typename T>
void ReadInputFromBinFile (char const * filename, char* data_ptr, size_t data_size) {
    ifstream file;
    file.open(filename, ios::in|ios::binary);
    if (file) {
      file.read(data_ptr, data_size*sizeof(T));
      file.close();
    } else {
      std::cout << "Input file not found.\n";
      exit(0);
    }
}

void InitData(queue* q, size_t npoints, tfloat **x1, tfloat **y1, tfloat **z1, tfloat **w1,
	      tfloat **x2, tfloat **y2, tfloat **z2, tfloat **w2, tfloat **rbins, tfloat **results_test) {

  tfloat *t_x1, *t_y1, *t_z1, *t_w1, *t_x2, *t_y2, *t_z2, *t_w2, *t_rbins, *t_results_test;

  /* Allocate aligned memory */
  t_x1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  t_y1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  t_z1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  t_w1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  t_x2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  t_y2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  t_z2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  t_w2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);

  t_rbins = (tfloat*)_mm_malloc(DEFAULT_NBINS * sizeof(tfloat), ALIGN_FACTOR);
  t_results_test = (tfloat*)_mm_malloc(npoints*(DEFAULT_NBINS-1) * sizeof(tfloat), ALIGN_FACTOR);

  if ( (t_x1 == NULL) || (t_y1 == NULL) || (t_z1 == NULL) || (t_w1 == NULL) ||
       (t_x2 == NULL) || (t_y2 == NULL) || (t_z2 == NULL) || (t_w2 == NULL)) {
    printf("Memory allocation failure\n");
    exit(-1);
  }

  ReadInputFromBinFile<tfloat> ("x1.bin", reinterpret_cast<char *>(t_x1), npoints);
  ReadInputFromBinFile<tfloat> ("y1.bin", reinterpret_cast<char *>(t_y1), npoints);
  ReadInputFromBinFile<tfloat> ("z1.bin", reinterpret_cast<char *>(t_z1), npoints);
  ReadInputFromBinFile<tfloat> ("w1.bin", reinterpret_cast<char *>(t_w1), npoints);
  ReadInputFromBinFile<tfloat> ("x2.bin", reinterpret_cast<char *>(t_x2), npoints);
  ReadInputFromBinFile<tfloat> ("y2.bin", reinterpret_cast<char *>(t_y2), npoints);
  ReadInputFromBinFile<tfloat> ("z2.bin", reinterpret_cast<char *>(t_z2), npoints);
  ReadInputFromBinFile<tfloat> ("w2.bin", reinterpret_cast<char *>(t_w2), npoints);
  ReadInputFromBinFile<tfloat> ("DEFAULT_RBINS_SQUARED.bin", reinterpret_cast<char *>(t_rbins), DEFAULT_NBINS);
  memset (t_results_test,0,npoints*(DEFAULT_NBINS-1) * sizeof(tfloat));

  tfloat *d_x1, *d_y1, *d_z1, *d_w1, *d_x2, *d_y2, *d_z2, *d_w2, *d_rbins, *d_results_test;

  d_x1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_y1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_z1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_w1 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_x2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_y2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_z2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_w2 = (tfloat*)malloc_device( npoints * sizeof(tfloat), *q);
  d_rbins = (tfloat*)malloc_device( DEFAULT_NBINS * sizeof(tfloat), *q);
  d_results_test = (tfloat*)malloc_device( npoints*(DEFAULT_NBINS-1) * sizeof(tfloat), *q);

  // copy data host to device
  q->memcpy(d_x1, t_x1, npoints * sizeof(tfloat));
  q->memcpy(d_y1, t_y1, npoints * sizeof(tfloat));
  q->memcpy(d_z1, t_z1, npoints * sizeof(tfloat));
  q->memcpy(d_w1, t_w1, npoints * sizeof(tfloat));
  q->memcpy(d_x2, t_x2, npoints * sizeof(tfloat));
  q->memcpy(d_y2, t_y2, npoints * sizeof(tfloat));
  q->memcpy(d_z2, t_z2, npoints * sizeof(tfloat));
  q->memcpy(d_w2, t_w2, npoints * sizeof(tfloat));
  q->memcpy(d_rbins, t_rbins, DEFAULT_NBINS * sizeof(tfloat));
  q->memcpy(d_results_test, t_results_test, npoints*(DEFAULT_NBINS-1) * sizeof(tfloat));

  q->wait();

  *x1 = d_x1;
  *y1 = d_y1;
  *z1 = d_z1;
  *w1 = d_w1;
  *x2 = d_x2;
  *y2 = d_y2;
  *z2 = d_z2;
  *w2 = d_w2;
  *rbins = d_rbins;
  *results_test = d_results_test;

  /* Free memory */
  _mm_free(t_x1);
  _mm_free(t_y1);
  _mm_free(t_z1);
  _mm_free(t_w1);
  _mm_free(t_x2);
  _mm_free(t_y2);
  _mm_free(t_z2);
  _mm_free(t_w2);
  _mm_free(t_rbins);
  _mm_free(t_results_test);
}

void ResetResult (size_t npoints, queue* q, tfloat* results_test) {
  tfloat* t_results_test = (tfloat*)_mm_malloc(npoints*(DEFAULT_NBINS-1) * sizeof(tfloat), ALIGN_FACTOR);
  memset (t_results_test,0,npoints*(DEFAULT_NBINS-1) * sizeof(tfloat));
  q->memcpy(results_test, t_results_test, npoints*(DEFAULT_NBINS-1) * sizeof(tfloat));
  q->wait();
  _mm_free(t_results_test);
}

/* Deallocate arrays */
void FreeData( queue* q, tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
	       tfloat *x2, tfloat *y2, tfloat *z2, tfloat *w2, tfloat *rbins, tfloat *results_test )
{
  free(x1,q->get_context());
  free(y1,q->get_context());
  free(z1,q->get_context());
  free(w1,q->get_context());
  free(x2,q->get_context());
  free(y2,q->get_context());
  free(z2,q->get_context());
  free(w2,q->get_context());
  free(rbins,q->get_context());
  free(results_test, q->get_context());
}
