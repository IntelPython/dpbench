#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED

#include "constants_header.h"

typedef struct {
  tfloat x;
  tfloat y;
  int cluster;
} Point;

typedef struct {
  tfloat x;
  tfloat y;
  tfloat x_sum;
  tfloat y_sum;
  int num_points;
} Centroid;

#endif
