#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

#include "kmeans.h"
#include "point.h"

#define REPEAT 1

#define ITERATIONS 30

void groupByCluster(queue* q,
		    Point* points,
		    Centroid* centroids,
		    int num_centroids,
		    size_t num_points
		    ) {

  Centroid* d_centroids = (Centroid*)malloc_device(num_centroids * sizeof(Centroid), *q);
  Point* d_points = (Point*)malloc_device(num_points * sizeof(Point), *q);

  q->memcpy(d_centroids, centroids, num_centroids * sizeof(Centroid));
  q->memcpy(d_points, points, num_points * sizeof(Point));

  q->submit([&](handler& h) {
      h.parallel_for<class theKernel>(range<1>{num_points}, [=](id<1> myID) {
  	  int i0 = myID[0];

  	  float minor_distance = -1.0;

  	  for (int i1 = 0; i1 < num_centroids; i1++) {
  	    float dx = d_points[i0].x - d_centroids[i1].x;
  	    float dy = d_points[i0].y - d_centroids[i1].y;
  	    float my_distance = cl::sycl::sqrt(dx*dx + dy*dy);
  	    if (minor_distance > my_distance || minor_distance == -1.0) {
  	      minor_distance = my_distance;
  	      d_points[i0].cluster = i1;
  	    }
  	  }
  	});
    });

  q->wait();

  q->memcpy(points, d_points, num_points * sizeof(Point));

  q->wait();

  free(d_centroids,q->get_context());
  free(d_points,q->get_context());
}


void calCentroidsSum(
    Point* points,
    Centroid* centroids,
    int num_centroids,
    size_t num_points
) {
#pragma omp parallel for simd
    for(int i = 0; i < num_centroids; i++) {
        centroids[i].x_sum = 0.0;
        centroids[i].y_sum = 0.0;
        centroids[i].num_points = 0.0;
    }

    for(int i = 0; i < num_points; i++) {
        int ci = points[i].cluster;
        centroids[ci].x_sum += points[i].x;
        centroids[ci].y_sum += points[i].y;
        centroids[ci].num_points += 1;
    }
}


void updateCentroids(
    Centroid* centroids,
    int num_centroids
) {
#pragma omp parallel for simd
	for(int i = 0; i < num_centroids; i++) {
	    if (centroids[i].num_points > 0) {
	        centroids[i].x = centroids[i].x_sum / centroids[i].num_points;
	        centroids[i].y = centroids[i].y_sum / centroids[i].num_points;
	    }
	}
}


void kmeans(queue* q,
	    Point* h_points,
	    Centroid* h_centroids,
	    size_t num_points,
	    int num_centroids
) {
    for(int i = 0; i < ITERATIONS; i++) {
      groupByCluster(q,
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
		    int NUMBER_OF_CENTROIDS
) {
    for (int i = 0; i < NUMBER_OF_CENTROIDS; i++) {
        printf("[x=%lf, y=%lf, x_sum=%lf, y_sum=%lf, num_points=%i]\n",
               centroids[i].x, centroids[i].y, centroids[i].x_sum,
               centroids[i].y_sum, centroids[i].num_points);
    }

    printf("--------------------------------------------------\n");
}


void runKmeans(queue* q,
	       Point* points,
	       Centroid* centroids,
	       size_t NUMBER_OF_POINTS,
	       int NUMBER_OF_CENTROIDS
) {

    for (int i = 0; i < REPEAT; i++) {
        for (int ci = 0; ci < NUMBER_OF_CENTROIDS; ci++) {
            centroids[ci].x = points[ci].x;
            centroids[ci].y = points[ci].y;
        }

        kmeans(q, points, centroids, NUMBER_OF_POINTS, NUMBER_OF_CENTROIDS);
    }
}
