/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __CONSTANTS_HEADER_H
#define __CONSTANTS_HEADER_H

#ifdef __DO_FLOAT__
    typedef float tfloat;
#else
    typedef double  tfloat;
#endif

#define ALIGN_FACTOR 64

#define SEED 7777777

#define XL      0.0f
#define XH      100.0f

struct point {
  tfloat x;
  tfloat y;
  tfloat z;
};

void InitData( int nopt, struct point* *x1, struct point* *x2, tfloat** distance_op );
void FreeData( struct point *x1, struct point *x2 );
void pairwise_distance( int nopt, struct point* x1, struct point* x2, tfloat* distance_op );

#endif // #ifndef __CONSTANTS_HEADER_H
