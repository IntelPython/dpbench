/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __CONSTANTS_HEADER_H
#define __CONSTANTS_HEADER_H

#ifdef __DO_FLOAT__
    typedef float tfloat;
    typedef unsigned int tint;
#else
    typedef double tfloat;
    typedef size_t tint;
#endif

#define ALIGN_FACTOR 64

#define SEED 7777777

#define XL      0.0f
#define XH      4.0f

#endif // #ifndef __CONSTANTS_HEADER_H
