/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#define _XOPEN_SOURCE
#define _DEFAULT_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <ia32intrin.h>
#include <fstream>

#include "euro_opt.h"

using namespace std;
using namespace cl::sycl;

tfloat RandRange( tfloat a, tfloat b, struct drand48_data *seed ) {
  double r;
  drand48_r(seed, &r);
  return r*(b-a) + a;
}

/*
// This function allocates arrays to hold input and output parameters
// for the Black-Scholes formula.
//     nopt - length of arrays
// Random input parameters
//     s0   - initial price
//     x    - strike price
//     t    - maturity
// Output arrays for call and put prices
//     vcall_compiler
//     vput_compiler
*/
void InitData( queue *q, size_t nopt, tfloat* *s0, tfloat* *x, tfloat* *t,
	       tfloat* *vcall_compiler, tfloat* *vput_compiler)
{
  tfloat *ts0, *tx, *tt, *tvcall_compiler, *tvput_compiler;
  size_t i;

  /* Allocate aligned memory */
  ts0             = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
  tx              = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
  tt              = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
  tvcall_compiler = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);
  tvput_compiler  = (tfloat*)_mm_malloc( nopt * sizeof(tfloat), ALIGN_FACTOR);

  if ( (ts0 == NULL) || (tx == NULL) || (tt == NULL) ||
       (tvcall_compiler == NULL) || (tvput_compiler == NULL) ) {
    printf("Memory allocation failure\n");
    exit(-1);
  }

  ifstream file;
  file.open("price.bin", ios::in|ios::binary);
  if (file) {
    file.read(reinterpret_cast<char *>(ts0), nopt*sizeof(tfloat));
    file.close();
  } else {
    std::cout << "Input file not found.\n";
    exit(0);
  }

  file.open("strike.bin", ios::in|ios::binary);
  if (file) {
    file.read(reinterpret_cast<char *>(tx), nopt*sizeof(tfloat));
    file.close();
  } else {
    std::cout << "Input file not found.\n";
    exit(0);
  }

  file.open("t.bin", ios::in|ios::binary);
  if (file) {
    file.read(reinterpret_cast<char *>(tt), nopt*sizeof(tfloat));
    file.close();
  } else {
    std::cout << "Input file not found.\n";
    exit(0);
  }

  for ( i = 0; i < nopt; i++ ){
    tvcall_compiler[i] = 0.0;
    tvput_compiler[i]  = 0.0;
  }

  tfloat *d_price, *d_strike, *d_t, *d_vcall, *d_vput;
  d_price = (tfloat*)malloc_device( nopt * sizeof(tfloat), *q);
  d_strike = (tfloat*)malloc_device( nopt * sizeof(tfloat), *q);
  d_t = (tfloat*)malloc_device( nopt * sizeof(tfloat), *q);
  d_vcall = (tfloat*)malloc_device( nopt * sizeof(tfloat), *q);
  d_vput = (tfloat*)malloc_device( nopt * sizeof(tfloat), *q);

  // copy data host to device
  q->memcpy(d_price, ts0, nopt * sizeof(tfloat));
  q->memcpy(d_strike, tx, nopt * sizeof(tfloat));
  q->memcpy(d_t, tt, nopt * sizeof(tfloat));
  q->memcpy(d_vcall, tvcall_compiler, nopt * sizeof(tfloat));
  q->memcpy(d_vput, tvput_compiler, nopt * sizeof(tfloat));
  q->wait();

  *s0 = d_price;
  *x  = d_strike;
  *t  = d_t;
  *vcall_compiler = d_vcall;
  *vput_compiler  = d_vput;

  /* Free memory */
  _mm_free(ts0);
  _mm_free(tx);
  _mm_free(tt);
  _mm_free(tvcall_compiler);
  _mm_free(tvput_compiler);
}

/* Deallocate arrays */
void FreeData( queue *q, tfloat *s0, tfloat *x, tfloat *t,
                   tfloat *vcall_compiler, tfloat *vput_compiler)
{
  /* Free memory */
  free(s0,q->get_context());
  free(x,q->get_context());
  free(t,q->get_context());
  free(vcall_compiler,q->get_context());
  free(vput_compiler,q->get_context());
}
