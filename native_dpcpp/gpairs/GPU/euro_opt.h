/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __EURO_OPT_BENCH_H
#define __EURO_OPT_BENCH_H

#include <CL/sycl.hpp>
using namespace cl::sycl;
using namespace std;

#ifdef __DO_FLOAT__
    typedef float tfloat;
#else
    typedef double  tfloat;
#endif

#define ALIGN_FACTOR 64

#define SEED 43

#define TL      1.0f
#define LBOX    500.0f
#define DEFAULT_NBINS 20

void InitData( queue* q, size_t npoints, tfloat **x1, tfloat **y1, tfloat **z1, tfloat **w1,
	      tfloat **x2, tfloat **y2, tfloat **z2, tfloat **w2, tfloat **rbins, tfloat **results_test);

void FreeData( queue* q, tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
	       tfloat *x2, tfloat *y2, tfloat *z2, tfloat *w2, tfloat *rbins, tfloat *results_test);

sycl::event gpairs_slm(
    sycl::queue *q,
    size_t n,
    tfloat *x0, tfloat *y0, tfloat *z0, tfloat *w0,
    tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
    size_t nbins, tfloat *rbins, tfloat *hist);

sycl::event gpairs_no_slm(
    sycl::queue *q,
    size_t n,
    tfloat *x0, tfloat *y0, tfloat *z0, tfloat *w0,
    tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
    size_t nbins, tfloat *rbins, tfloat *hist);

sycl::event gpairs_orig(
    sycl::queue *q,
    size_t n,
    tfloat *x0, tfloat *y0, tfloat *z0, tfloat *w0,
    tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
    size_t nbins, tfloat *rbins, tfloat *hist);

void gpairs_host(
    sycl::queue *q,
    size_t n,
    tfloat *x0, tfloat *y0, tfloat *z0, tfloat *w0,
    tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
    size_t nbins, tfloat *rbins, tfloat *hist);

void ResetResult (queue* q, tfloat* results_test);

#endif // #ifndef __EURO_OPT_BENCH_H
