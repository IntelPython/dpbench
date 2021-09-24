/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <omp.h>
#include <mathimf.h>
#include "constants_header.h"

#ifdef __DO_FLOAT__
#   define SQRT(x)     sqrtf(x)
#else
#   define SQRT(x)     sqrt(x)
#endif

void l2_distance( int nopt, tfloat * x1, tfloat * x2, tfloat* distance_op ) {
  int i;
  tfloat a, b, sum = 0;

#pragma omp parallel for simd shared(x1, x2) reduction(+: sum)
  for ( i = 0; i < nopt; i++ ) {
    a = x1[i] - x2[i];
    b = a * a;
    sum += b;
  }
  *distance_op = SQRT(sum);
}
