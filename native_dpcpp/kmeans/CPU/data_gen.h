/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __DATA_GEN_H
#define __DATA_GEN_H

#include "point.h"
#include <CL/sycl.hpp>

void InitData( cl::sycl::queue *q, size_t nopt, int ncentroids, Point** points, Centroid** centroids );
void FreeData( cl::sycl::queue *q, Point *pts, Centroid * cents );

#endif // #ifndef __DATA_GEN_H
