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

#include <CL/sycl.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "rdtsc.h"

#define DATADIM 16
#define NEAREST_NEIGHS 5
#define NUM_CLASSES 3

using namespace cl::sycl;

template <typename FpTy, typename IntTy> class theKernel;

template <typename FpTy> struct neighbors
{
    FpTy dist;
    FpTy label;
};

template <typename FpTy, typename IntTy>
void knn_impl(sycl::queue &q,
              FpTy *d_train,
              size_t *d_train_labels,
              FpTy *d_test,
              size_t k,
              size_t classes_num,
              size_t train_size,
              size_t test_size,
              IntTy *d_predictions,
              FpTy *d_votes_to_classes,
              size_t data_dim)
{
    q.submit([&](sycl::handler &h) {
        h.parallel_for<theKernel<FpTy, IntTy>>(
            sycl::range<1>{test_size}, [=](sycl::id<1> myID) {
                size_t i = myID[0];

                // here k has to be 5 in order to match with numpy no. of
                // neighbors
                struct neighbors<FpTy> queue_neighbors[5];

                // count distances
                for (size_t j = 0; j < k; ++j) {
                    FpTy distance = 0.0;
                    for (std::size_t jj = 0; jj < data_dim; ++jj) {
                        FpTy diff = d_train[j * data_dim + jj] -
                                    d_test[i * data_dim + jj];
                        distance += diff * diff;
                    }

                    FpTy dist = sqrt(distance);

                    queue_neighbors[j].dist = dist;
                    queue_neighbors[j].label = d_train_labels[j];
                }

                // sort queue
                for (size_t j = 0; j < k; ++j) {
                    // push queue
                    FpTy new_distance = queue_neighbors[j].dist;
                    FpTy new_neighbor_label = queue_neighbors[j].label;
                    size_t index = j;
                    while (index > 0 &&
                           new_distance < queue_neighbors[index - 1].dist)
                    {
                        queue_neighbors[index].dist =
                            queue_neighbors[index - 1].dist;
                        queue_neighbors[index].label =
                            queue_neighbors[index - 1].label;
                        index--;

                        queue_neighbors[index].dist = new_distance;
                        queue_neighbors[index].label = new_neighbor_label;
                    }
                }

                for (size_t j = k; j < train_size; ++j) {
                    FpTy distance = 0.0;
                    for (std::size_t jj = 0; jj < data_dim; ++jj) {
                        FpTy diff = d_train[j * data_dim + jj] -
                                    d_test[i * data_dim + jj];
                        distance += diff * diff;
                    }

                    FpTy dist = sqrt(distance);

                    if (dist < queue_neighbors[k - 1].dist) {
                        queue_neighbors[k - 1].dist = dist;
                        queue_neighbors[k - 1].label = d_train_labels[j];

                        // push queue
                        FpTy new_distance = queue_neighbors[k - 1].dist;
                        FpTy new_neighbor_label = queue_neighbors[k - 1].label;
                        size_t index = k - 1;

                        while (index > 0 &&
                               new_distance < queue_neighbors[index - 1].dist)
                        {
                            queue_neighbors[index].dist =
                                queue_neighbors[index - 1].dist;
                            queue_neighbors[index].label =
                                queue_neighbors[index - 1].label;
                            index--;

                            queue_neighbors[index].dist = new_distance;
                            queue_neighbors[index].label = new_neighbor_label;
                        }
                    }
                }

                // simple vote
                for (size_t j = 0; j < k; ++j) {
                    d_votes_to_classes[i * classes_num +
                                       size_t(queue_neighbors[j].label)]++;
                }

                IntTy max_ind = 0;
                FpTy max_value = 0.0;

                for (IntTy j = 0; j < (IntTy)classes_num; ++j) {
                    if (d_votes_to_classes[i * classes_num + j] > max_value) {
                        max_value = d_votes_to_classes[i * classes_num + j];
                        max_ind = j;
                    }
                }
                d_predictions[i] = max_ind;
            });
    });

    q.wait();
}

auto read_data_x(size_t data_size, std::string filename)
{

    auto n = data_size * DATADIM;
    auto data = std::make_unique<double[]>(n);

    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file) {
        file.read(reinterpret_cast<char *>(data.get()), n * sizeof(double));
        file.close();
    }
    else {
        std::cout << "Input file not found.\n";
        exit(0);
    }

    return data;
}

auto read_data_y(size_t data_size, std::string filename)
{

    auto n = data_size;
    auto data = std::make_unique<size_t[]>(n);

    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file) {
        file.read(reinterpret_cast<char *>(data.get()), n * sizeof(size_t));
        file.close();
    }
    else {
        std::cout << "Input file not found.\n";
        exit(0);
    }

    return data;
}

int main(int argc, char *argv[])
{
    double t1 = 0, t2 = 0;

    size_t nPoints_train = pow(2, 10);
    size_t nPoints = 16777216;

    queue q;

    auto data_train_ptr = read_data_x(nPoints_train, "x_train.bin");
    double *data_train = data_train_ptr.get();

    auto train_labels_ptr = read_data_y(nPoints_train, "y_train.bin");
    size_t *train_labels = train_labels_ptr.get();

    auto data_test_ptr = read_data_x(nPoints, "x_test.bin");
    double *data_test = data_test_ptr.get();

    // size_t *predictions = new size_t[nPoints];

    double *d_votes_to_classes =
        (double *)malloc_device(nPoints * NUM_CLASSES * sizeof(double), q);

    double *d_train =
        (double *)malloc_device(nPoints_train * DATADIM * sizeof(double), q);
    size_t *d_train_labels =
        (size_t *)malloc_device(nPoints_train * sizeof(size_t), q);
    double *d_test =
        (double *)malloc_device(nPoints * DATADIM * sizeof(double), q);
    size_t *d_predictions =
        (size_t *)malloc_device(nPoints * sizeof(size_t), q);

    // copy data host to device
    q.memcpy(d_train, data_train, nPoints_train * DATADIM * sizeof(double));
    q.memcpy(d_train_labels, train_labels, nPoints_train * sizeof(size_t));
    q.memcpy(d_test, data_test, nPoints * DATADIM * sizeof(double));
    q.wait();

    /* Warm up cycle */
    knn_impl<double, size_t>(q, d_train, d_train_labels, d_test, NEAREST_NEIGHS,
                             NUM_CLASSES, nPoints_train, nPoints, d_predictions,
                             d_votes_to_classes, DATADIM);

    t1 = timer_rdtsc();
    knn_impl<double, size_t>(q, d_train, d_train_labels, d_test, NEAREST_NEIGHS,
                             NUM_CLASSES, nPoints_train, nPoints, d_predictions,
                             d_votes_to_classes, DATADIM);
    t2 = timer_rdtsc();

    // copy result device to host
    // q.memcpy(predictions, d_predictions, nPoints * sizeof(size_t));
    // q.wait();

    // free(d_train, q.get_context());
    // free(d_test, q.get_context());
    // free(d_train_labels, q.get_context());
    // free(d_predictions, q.get_context());
    // free(d_votes_to_classes, q.get_context());

    double time = ((double)(t2 - t1) / getHz());

    printf("ERF: Native-C-VML: Size: %ld Time: %.6lf\n", nPoints, time);
    // fflush(stdout);

    // std::ofstream file;
    // file.open("predictions.bin", std::ios::out|std::ios::binary);
    // if (file) {
    //   file.write(reinterpret_cast<char *>(predictions),
    //   nPoints*sizeof(double)); file.close();
    // } else {
    //   std::cout << "Unable to open output file.\n";
    // }

    // delete[] predictions;

    return 0;
}
