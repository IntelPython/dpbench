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
#define XH      1.0f

void InitData( int nopt, tfloat* *x1, tfloat* *x2, tfloat* distance_op );
void FreeData( tfloat *x1, tfloat *x2 );
void l2_distance( int nopt,tfloat* x1, tfloat* x2,tfloat* distance_op );

#endif // #ifndef __CONSTANTS_HEADER_H
