/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __DATA_GEN_H
#define __DATA_GEN_H

#include "point.h"

void InitData( size_t nopt, int ncentroids, Point** points, Centroid** centroids );
void FreeData( Point *pts, Centroid * cents );

#endif // #ifndef __DATA_GEN_H
