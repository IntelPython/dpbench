/*
Copyright (c) 2020, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "knn.h"

void push_queue(struct neighbors* queue, double new_distance, size_t new_label, int index)
{

    while (index > 0 && new_distance < queue[index-1].dist)
    {
        queue[index] = queue[index - 1];
        --index;
        queue[index].dist = new_distance;
	queue[index].label = new_label;
    }

}

void sort_queue(struct neighbors* queue)
{
    for (int i = 1; i < NEAREST_NEIGHS; i++)
    {
      push_queue(queue, queue[i].dist, queue[i].label, i);
    }
}

double euclidean_dist(double* x1, size_t jj, double* x2, size_t ii)
{
    double distance = 0.0;
    for (std::size_t i = 0; i < DATADIM; ++i)
    {
        double diff = x1[jj*DATADIM + i] - x2[ii*DATADIM + i];
        distance += diff * diff;
    }

    double result = sqrt(distance);

    return result;
}

size_t simple_vote(struct neighbors* neighbors)
{
  size_t votes_to_classes[NUM_CLASSES] = {0};

  for (int i = 0; i < NEAREST_NEIGHS; ++i) {
    votes_to_classes[neighbors[i].label]++;
  }

  size_t max_ind = 0, max_value = 0;

  for (int i = 0; i < NUM_CLASSES; ++i) {
    if (votes_to_classes[i] > max_value) {
      max_value = votes_to_classes[i];
      max_ind = i;
    }
  }

  return max_ind;
}

void run_knn(queue* q, double* train, size_t *train_labels, double* test, size_t train_nrows, size_t test_size, size_t *predictions) {
  double *d_train, *d_test; size_t *d_train_labels, *d_predictions;

  d_train = (double*)malloc_device( train_nrows * DATADIM * sizeof(double), *q);
  d_train_labels = (size_t*)malloc_device( train_nrows * sizeof(size_t), *q);
  d_test = (double*)malloc_device( test_size * DATADIM * sizeof(double*), *q);
  d_predictions = (size_t*)malloc_device( test_size * sizeof(size_t), *q);

  // copy data host to device
  q->memcpy(d_train, train, train_nrows * DATADIM * sizeof(double));
  q->memcpy(d_train_labels, train_labels, train_nrows * sizeof(size_t));
  q->memcpy(d_test, test, test_size * DATADIM * sizeof(double));
  
  q->submit([&](handler& h) {
      h.parallel_for<class theKernel>(range<1>{test_size}, [=](id<1> myID) {
	  size_t i = myID[0];
	  //std::array<std::pair<double, size_t>, NEAREST_NEIGHS> queue_neighbors;
	  struct neighbors queue_neighbors[NEAREST_NEIGHS] = {{ 0 }};
      
	  //count distances
	  for (int j = 0; j < NEAREST_NEIGHS; ++j) {
	    queue_neighbors[j].dist = euclidean_dist(d_train, j, d_test, i);
	    queue_neighbors[j].label = d_train_labels[j];
	  }

	  sort_queue(queue_neighbors);

	  for (int j = NEAREST_NEIGHS; j < train_nrows; ++j) {
	    double dist = euclidean_dist(d_train, j, d_test, i);
	    //auto new_neighbor = std::make_pair(dist, train_labels[j]);

	    if (dist < queue_neighbors[NEAREST_NEIGHS-1].dist) {
	      //queue_neighbors[NEAREST_NEIGHS-1] = new_neighbor;
	      queue_neighbors[NEAREST_NEIGHS-1].dist = dist;
	      queue_neighbors[NEAREST_NEIGHS-1].label = d_train_labels[j];
	  
	      push_queue(queue_neighbors, dist, d_train_labels[j], NEAREST_NEIGHS-1);
	    }
	  }
	  d_predictions[i] = simple_vote(queue_neighbors);
	});
    }).wait();

  q->memcpy(predictions, d_predictions, test_size * sizeof(size_t));

  q->wait();

  free(d_train, q->get_context());
  free(d_test, q->get_context());
  free(d_train_labels, q->get_context());
  free(d_predictions, q->get_context());
}
