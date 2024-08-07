/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __CONSTANTS_HEADER_H
#define __CONSTANTS_HEADER_H

#include <CL/sycl.hpp>

#ifdef __DO_FLOAT__
typedef float tfloat;
#else
typedef double tfloat;
#endif

#define ALIGN_FACTOR 64

void InitData(cl::sycl::queue &q,
              size_t nopt,
              size_t ndims,
              tfloat **a,
              tfloat **distance_op);
void FreeData(cl::sycl::queue &q, tfloat *a, tfloat *distance_op);
void l2_norm_impl(cl::sycl::queue &q,
                  size_t nopt,
                  size_t ndims,
                  tfloat *a,
                  tfloat *distance_op);

#endif // #ifndef __CONSTANTS_HEADER_H
