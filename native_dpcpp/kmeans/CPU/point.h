#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED

typedef struct {
    float x;
    float y;
    int cluster;
} Point;

typedef struct {
    float x;
    float y;
    float x_sum;
    float y_sum;
    int num_points;
} Centroid;

#endif
