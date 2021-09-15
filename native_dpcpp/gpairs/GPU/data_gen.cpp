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
      std::cout << "Input file - " << filename << " not found.\n";
      exit(0);
    }
}

void WriteOutputToTextFile (char const * filename, tfloat* data_ptr, size_t data_size) {
    ofstream file;
    file.open(filename, ios::out);
    if (file) {
      for (size_t i = 0; i < data_size; i++) {
	file << *data_ptr << std::endl;
      }
      file.close();
    } else {
      std::cout << "Input file - " << filename << " not found.\n";
      exit(0);
    }
}

void InitData(queue* q, size_t npoints, tfloat **x1, tfloat **y1, tfloat **z1, tfloat **w1,
	      tfloat **x2, tfloat **y2, tfloat **z2, tfloat **w2, tfloat **rbins, tfloat **results_test) {
  /* Allocate aligned memory */
  *x1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *y1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *z1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *w1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *x2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *y2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *z2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
  *w2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);

  *rbins = (tfloat*)_mm_malloc(DEFAULT_NBINS * sizeof(tfloat), ALIGN_FACTOR);
  *results_test = (tfloat*)_mm_malloc((DEFAULT_NBINS-1) * sizeof(tfloat), ALIGN_FACTOR);  

  if ( (*x1 == NULL) || (*y1 == NULL) || (*z1 == NULL) || (*w1 == NULL) ||
       (*x2 == NULL) || (*y2 == NULL) || (*z2 == NULL) || (*w2 == NULL)) {
    printf("Memory allocation failure\n");
    exit(-1);
  }

  ReadInputFromBinFile<tfloat> ("x1.bin", reinterpret_cast<char *>(*x1), npoints);
  ReadInputFromBinFile<tfloat> ("y1.bin", reinterpret_cast<char *>(*y1), npoints);
  ReadInputFromBinFile<tfloat> ("z1.bin", reinterpret_cast<char *>(*z1), npoints);
  ReadInputFromBinFile<tfloat> ("w1.bin", reinterpret_cast<char *>(*w1), npoints);
  ReadInputFromBinFile<tfloat> ("x2.bin", reinterpret_cast<char *>(*x2), npoints);
  ReadInputFromBinFile<tfloat> ("y2.bin", reinterpret_cast<char *>(*y2), npoints);
  ReadInputFromBinFile<tfloat> ("z2.bin", reinterpret_cast<char *>(*z2), npoints);
  ReadInputFromBinFile<tfloat> ("w2.bin", reinterpret_cast<char *>(*w2), npoints);
  ReadInputFromBinFile<tfloat> ("DEFAULT_RBINS_SQUARED.bin", reinterpret_cast<char *>(*rbins), DEFAULT_NBINS);
  memset (*results_test,0,(DEFAULT_NBINS-1) * sizeof(tfloat));
}

void ResetResult (queue* q, tfloat* results_test) {
  memset (results_test,0,(DEFAULT_NBINS-1) * sizeof(tfloat));
}

/* Deallocate arrays */
void FreeData( queue* q, tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
	       tfloat *x2, tfloat *y2, tfloat *z2, tfloat *w2, tfloat *rbins, tfloat *results_test )
{
    /* Free memory */
    _mm_free(x1);
    _mm_free(y1);
    _mm_free(z1);
    _mm_free(w1);
    _mm_free(x2);
    _mm_free(y2);
    _mm_free(z2);
    _mm_free(w2);
    _mm_free(rbins);
    _mm_free(results_test);
}
