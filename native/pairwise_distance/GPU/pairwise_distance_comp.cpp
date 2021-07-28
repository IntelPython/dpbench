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

void pairwise_distance( size_t nopt, struct point * p1, struct point * p2, tfloat* distance_op ) {
  size_t i, j;
  tfloat tmp;
#pragma omp target teams distribute					\
  parallel for simd shared(p1, p2, distance_op) private(tmp,j)
  for (i = 0; i < nopt; i++) {
    for (j = 0; j < nopt; j++) {
      tmp = p1[i].x - p2[j].x;
      tfloat d = tmp * tmp;

      tmp = p1[i].y - p2[j].y;
      d += tmp * tmp;

      tmp = p1[i].z - p2[j].z;
      d += tmp * tmp;

      if (d != 0.0) {
	distance_op[i*nopt + j] = sqrt(d);
      }
    }
  }   
}
