#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED

#include "point.h"

/* void groupByCluster(Point* points, Centroid* centroids, int num_centroids); */

/* void sumPointsCluster(Point* points, Centroid* centroids, int num_centroids); */

/* void updateCentroids(Centroid* centroids); */

/* void kmeans(Point* h_points, Centroid* h_centroids, int num_points, int num_centroids); */

void runKmeans(cl::sycl::queue* q,
	       Point* points, 
	       Centroid* centroids,
	       size_t NUMBER_OF_POINTS,
	       int NUMBER_OF_CENTROIDS
	       );

void printCentroids(
		    Centroid* centroids,
		    int NUMBER_OF_CENTROIDS
		    );

#endif
