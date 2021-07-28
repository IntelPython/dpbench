#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#include "kmeans.h"
#include "point.h"

#define REPEAT 1

#define ITERATIONS 30

float distance(Point* p, Centroid* c) {
    float dx = p->x - c->x;
    float dy = p->y - c->y;
    return sqrtf(dx*dx + dy*dy);
}

void groupByCluster(
    Point* points,
    Centroid* centroids,
    size_t num_centroids, 
    size_t num_points
) {
#pragma omp parallel for simd
	for(size_t i0 = 0; i0 < num_points; i0++) {
		float minor_distance = -1.0;

		for (size_t i1 = 0; i1 < num_centroids; i1++) {
			float my_distance = distance(&points[i0], &centroids[i1]);
			if (minor_distance > my_distance || minor_distance == -1.0) {
				minor_distance = my_distance;
				points[i0].cluster = i1;
			}
		}
	}
}


void calCentroidsSum(
    Point* points, 
    Centroid* centroids,
    size_t num_centroids, 
    size_t num_points
) {
#pragma omp parallel for simd
    for(size_t i = 0; i < num_centroids; i++) {
        centroids[i].x_sum = 0.0;
        centroids[i].y_sum = 0.0;
        centroids[i].num_points = 0.0;
    }

    for(size_t i = 0; i < num_points; i++) {
        size_t ci = points[i].cluster;
        centroids[ci].x_sum += points[i].x;
        centroids[ci].y_sum += points[i].y;
        centroids[ci].num_points += 1;
    }
}


void updateCentroids(
    Centroid* centroids, 
    size_t num_centroids
) {
#pragma omp parallel for simd
	for(size_t i = 0; i < num_centroids; i++) {
	    if (centroids[i].num_points > 0) {
	        centroids[i].x = centroids[i].x_sum / centroids[i].num_points;
	        centroids[i].y = centroids[i].y_sum / centroids[i].num_points;
	    }
	}
}


void kmeans(
    Point* h_points,
    Centroid* h_centroids, 
    size_t num_points,
    size_t num_centroids
) {
    for(size_t i = 0; i < ITERATIONS; i++) {
        groupByCluster(
            h_points, 
            h_centroids,
            num_centroids, 
            num_points
        );
        
        calCentroidsSum(
            h_points, 
            h_centroids,
            num_centroids, 
            num_points
        );

        updateCentroids(
            h_centroids, 
            num_centroids
        );
    }
}

void printCentroids(
		    Centroid* centroids,
		    size_t NUMBER_OF_CENTROIDS
) {
    for (size_t i = 0; i < NUMBER_OF_CENTROIDS; i++) {
        printf("[x=%lf, y=%lf, x_sum=%lf, y_sum=%lf, num_points=%i]\n", 
               centroids[i].x, centroids[i].y, centroids[i].x_sum,
               centroids[i].y_sum, centroids[i].num_points);
    }

    printf("--------------------------------------------------\n");
}


void runKmeans(
    Point* points, 
    Centroid* centroids,
    size_t NUMBER_OF_POINTS,
    size_t NUMBER_OF_CENTROIDS
) {

    for (size_t i = 0; i < REPEAT; i++) {
        for (size_t ci = 0; ci < NUMBER_OF_CENTROIDS; ci++) {
            centroids[ci].x = points[ci].x;
            centroids[ci].y = points[ci].y;
        }

        kmeans(points, centroids, NUMBER_OF_POINTS, NUMBER_OF_CENTROIDS);
        
        //if (i + 1 == REPEAT) {
	//   printCentroids(centroids);
        //}
    }
}
