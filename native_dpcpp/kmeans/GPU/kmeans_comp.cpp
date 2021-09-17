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
		    size_t num_centroids, 
		    size_t num_points
		    ) {
  q->submit([&](handler& h) {
      h.parallel_for<class theKernel>(range<1>{num_points}, [=](id<1> myID) {
  	  size_t i0 = myID[0];
  
  	  tfloat minor_distance = -1.0;

  	  for (size_t i1 = 0; i1 < num_centroids; i1++) {
  	    tfloat dx = points[i0].x - centroids[i1].x;
  	    tfloat dy = points[i0].y - centroids[i1].y;
  	    tfloat my_distance = cl::sycl::sqrt(dx*dx + dy*dy);
  	    if (minor_distance > my_distance || minor_distance == -1.0) {
  	      minor_distance = my_distance;
  	      points[i0].cluster = i1;
  	    }
  	  }
  	});
    });

  //q->wait();
}


void calCentroidsSum(queue* q,
		     Point* points,
		     Centroid* centroids,
		     size_t num_centroids,
		     size_t num_points
) {

  q->submit([&](handler& h) {
      h.parallel_for<class theKernel_1>(range<1>{num_centroids}, [=](id<1> myID_k1) {
	  
  	  size_t i = myID_k1[0];
  	  centroids[i].x_sum = 0.0;
  	  centroids[i].y_sum = 0.0;
  	  centroids[i].num_points = 0.0;  

  	});
    });
  
  q->wait();

  q->submit([&](handler& h) {
      h.parallel_for<class theKernel_2>(range<1>{num_points}, [=](id<1> myID) {
	  
  	  size_t i = myID[0];
  	  size_t ci = points[i].cluster;

  	  sycl::ext::oneapi::atomic_ref<tfloat, sycl::ext::oneapi::memory_order::relaxed, sycl::ext::oneapi::memory_scope::system,
  		     access::address_space::global_space> centroid_x_sum(centroids[ci].x_sum);
  	  centroid_x_sum += points[i].x;

  	  sycl::ext::oneapi::atomic_ref<tfloat, sycl::ext::oneapi::memory_order::relaxed, sycl::ext::oneapi::memory_scope::system,
  		     access::address_space::global_space> centroid_y_sum(centroids[ci].y_sum);	  
  	  centroid_y_sum += points[i].y;

  	  sycl::ext::oneapi::atomic_ref<tint, sycl::ext::oneapi::memory_order::relaxed, sycl::ext::oneapi::memory_scope::system,
  		     access::address_space::global_space> centroid_num_points(centroids[ci].num_points);	  
  	  centroid_num_points += 1;

  	});
    });

  q->wait();
}


void updateCentroids(queue* q,
		     Centroid* centroids, 
		     size_t num_centroids) {

  q->submit([&](handler& h) {
      h.parallel_for<class theKernel_uc>(range<1>{num_centroids}, [=](id<1> myID) {
	  
  	  size_t i = myID[0];
  	  if (centroids[i].num_points > 0) {
  	    centroids[i].x = centroids[i].x_sum / centroids[i].num_points;
  	    centroids[i].y = centroids[i].y_sum / centroids[i].num_points;
  	  }

  	});
    });
  
  q->wait();  
}


void kmeans(queue* q,
	    Point* h_points,
	    Centroid* h_centroids, 
	    size_t num_points,
	    size_t num_centroids
) {
    for(size_t i = 0; i < ITERATIONS; i++) {
      groupByCluster(q,
		     h_points, 
		     h_centroids,
		     num_centroids, 
		     num_points
		     );
        
      calCentroidsSum(q,
		      h_points, 
		      h_centroids,
		      num_centroids, 
		      num_points
		      );

      updateCentroids(q,
		      h_centroids, 
		      num_centroids
		      );
    }
}

void printCentroids(Centroid* centroids,
		    size_t NUMBER_OF_CENTROIDS
) {
    for (size_t i = 0; i < NUMBER_OF_CENTROIDS; i++) {
        printf("[x=%lf, y=%lf, x_sum=%lf, y_sum=%lf, num_points=%lu]\n", 
               centroids[i].x, centroids[i].y, centroids[i].x_sum,
               centroids[i].y_sum, centroids[i].num_points);
    }

    printf("--------------------------------------------------\n");
}


void runKmeans(queue* q,
	       Point* points, 
	       Centroid* centroids,
	       size_t NUMBER_OF_POINTS,
	       size_t NUMBER_OF_CENTROIDS) {
  for (size_t i = 0; i < REPEAT; i++) {

    q->submit([&](handler& h) {
    	h.parallel_for<class theKernel_km>(range<1>{NUMBER_OF_CENTROIDS}, [=](id<1> myID) {
	  
    	    size_t ci = myID[0];
    	    centroids[ci].x = points[ci].x;
    	    centroids[ci].y = points[ci].y;

    	  });
      });
    q->wait();
      
    kmeans(q, points, centroids, NUMBER_OF_POINTS, NUMBER_OF_CENTROIDS);
  }

  q->wait();
}
