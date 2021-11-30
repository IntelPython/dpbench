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

void push_queue(struct neighbors *queue, double new_distance, size_t new_label, int index)
{

  while (index > 0 && new_distance < queue[index - 1].dist)
  {
    queue[index] = queue[index - 1];
    --index;

    queue[index].dist = new_distance;
    queue[index].label = new_label;
  }
}

void sort_queue(struct neighbors *queue)
{
  for (int i = 1; i < NEAREST_NEIGHS; i++)
  {
    push_queue(queue, queue[i].dist, queue[i].label, i);
  }
}

double euclidean_dist(double *x1, size_t jj, double *x2, size_t ii)
{
  double distance = 0.0;
  for (std::size_t i = 0; i < DATADIM; ++i)
  {
    double diff = x1[jj * DATADIM + i] - x2[ii * DATADIM + i];
    distance += diff * diff;
  }

  double result = sqrt(distance);

  return result;
}

size_t simple_vote(struct neighbors *neighbors)
{
  size_t votes_to_classes[NUM_CLASSES] = {0};

  for (int i = 0; i < NEAREST_NEIGHS; ++i)
  {
    votes_to_classes[neighbors[i].label]++;
  }

  size_t max_ind = 0, max_value = 0;

  for (int i = 0; i < NUM_CLASSES; ++i)
  {
    if (votes_to_classes[i] > max_value)
    {
      max_value = votes_to_classes[i];
      max_ind = i;
    }
  }

  return max_ind;
}

void run_knn(queue *q, double *train, size_t *train_labels, double *test, size_t train_nrows, size_t test_size,
size_t *predictions, double *votes_to_classes)
{
  double *d_votes_to_classes = (double *)malloc_shared(test_size * NUM_CLASSES * sizeof(double), *q);
  //struct neighbors *d_queue_neighbors_lst = (struct neighbors *)malloc_shared(test_size * NEAREST_NEIGHS * sizeof(struct neighbors), *q);

  double *d_train = (double *)malloc_shared(train_nrows * DATADIM * sizeof(double), *q);
  size_t *d_train_labels = (size_t *)malloc_shared(train_nrows * sizeof(size_t), *q);
  double *d_test = (double *)malloc_shared(test_size * DATADIM * sizeof(double), *q);
  size_t *d_predictions = (size_t *)malloc_shared(test_size * sizeof(size_t), *q);

  // copy data host to device
  q->memcpy(d_train, train, train_nrows * DATADIM * sizeof(double));
  q->memcpy(d_train_labels, train_labels, train_nrows * sizeof(size_t));
  q->memcpy(d_test, test, test_size * DATADIM * sizeof(double));
  q->memcpy(d_votes_to_classes, votes_to_classes, test_size * NUM_CLASSES * sizeof(double));
  //q->memcpy(d_predictions, predictions, test_size * sizeof(size_t));
  //q->memcpy(d_queue_neighbors_lst, queue_neighbors_lst, test_size * NEAREST_NEIGHS * sizeof(struct neighbors));
  q->wait();

  q->submit([&](handler &h){
      h.parallel_for<class theKernel>(range<1>{test_size}, [=](id<1> myID) {
	  size_t i = myID[0];
	  struct neighbors queue_neighbors[NEAREST_NEIGHS];
	  //struct neighbors* queue_neighbors = &d_queue_neighbors_lst[i*NEAREST_NEIGHS];

	  //count distances
	  for (int j = 0; j < NEAREST_NEIGHS; ++j) {
	    double distance = 0.0;
	    for (std::size_t jj = 0; jj < DATADIM; ++jj) {
	      double diff = d_train[j * DATADIM + jj] - d_test[i * DATADIM + jj];
	      distance += diff * diff;
	    }

	    double dist = sqrt(distance);

	    queue_neighbors[j].dist = dist;
	    queue_neighbors[j].label = d_train_labels[j];
	  }

	  //sort queue
	  for (int j = 0; j < NEAREST_NEIGHS; ++j) {
	    // push queue
	    double new_distance = queue_neighbors[j].dist;
	    double new_neighbor_label = queue_neighbors[j].label;
	    int index = j;
	    while (index > 0 && new_distance < queue_neighbors[index-1].dist ) {
	      queue_neighbors[index] = queue_neighbors[index-1];
	      index--;

	      queue_neighbors[index].dist = new_distance;
	      queue_neighbors[index].label = new_neighbor_label;
	    }
	  }

	  for (int j = NEAREST_NEIGHS; j < train_nrows; ++j) {
	    double distance = 0.0;
	    for (std::size_t jj = 0; jj < DATADIM; ++jj) {
	      double diff = d_train[j * DATADIM + jj] - d_test[i * DATADIM + jj];
	      distance += diff * diff;
	    }

	    double dist = sqrt(distance);

	    if (dist < queue_neighbors[NEAREST_NEIGHS-1].dist) {
	      queue_neighbors[NEAREST_NEIGHS-1].dist = dist;
	      queue_neighbors[NEAREST_NEIGHS-1].label = d_train_labels[j];

	      //push queue
	      double new_distance = queue_neighbors[NEAREST_NEIGHS-1].dist;
	      double new_neighbor_label = queue_neighbors[NEAREST_NEIGHS-1].label;
	      int index = NEAREST_NEIGHS-1;

	      while (index > 0 && new_distance < queue_neighbors[index-1].dist) {
		queue_neighbors[index] = queue_neighbors[index-1];
		index--;

		queue_neighbors[index].dist = new_distance;
		queue_neighbors[index].label = new_neighbor_label;
	      }
	    }
	  }

	  // simple vote
	  for (int j = 0; j < NEAREST_NEIGHS; ++j) {
	    d_votes_to_classes[i*NUM_CLASSES + queue_neighbors[j].label]++;
	  }

	  int max_ind = 0;
	  double max_value = 0.0;

	  for (int j = 0; j < NUM_CLASSES; ++j) {
	    if (d_votes_to_classes[i*NUM_CLASSES + j] > max_value ) {
	      max_value = d_votes_to_classes[i*NUM_CLASSES + j];
	      max_ind = j;
	    }
	  }
	  d_predictions[i] =  max_ind;
	});
    }).wait();

  // copy data device to host
  q->memcpy(predictions, d_predictions, test_size * sizeof(size_t));
  // q->memcpy(train, d_train, train_nrows * DATADIM * sizeof(double));
  // q->memcpy(train_labels, d_train_labels, train_nrows * sizeof(size_t));
  // q->memcpy(test, test, d_test_size * DATADIM * sizeof(double));
  // q->memcpy(votes_to_classes, d_votes_to_classes, test_size * NUM_CLASSES * sizeof(double));
  // q->memcpy(queue_neighbors_lst, d_queue_neighbors_lst, test_size * NEAREST_NEIGHS * sizeof(struct neighbors));


  q->wait();

  free(d_train, q->get_context());
  free(d_test, q->get_context());
  free(d_train_labels, q->get_context());
  free(d_predictions, q->get_context());
  //free(d_queue_neighbors_lst, q->get_context());
  free(d_votes_to_classes, q->get_context());
}
