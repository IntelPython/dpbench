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


#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <omp.h>


using dtype = double;
const size_t classesNum = 3;
const size_t dataDim = 16;
const size_t k = 5; // Define the number of nearest neighbors

void push_queue(std::array<std::pair<dtype, size_t>, k> & queue, std::pair<dtype, size_t> new_distance, int index = k-1)
{

    while (index > 0 && new_distance.first < queue[index-1].first)
    {
        queue[index] = queue[index - 1];
        --index;

        queue[index] = new_distance;
    }

}

void sort_queue(std::array<std::pair<dtype, size_t>, k> &queue)
{
    for (int i = 1; i < queue.size(); i++)
    {
        push_queue(queue, queue[i], i);
    }
}


dtype euclidean_dist(std::array<dtype, dataDim>& x1, std::array<dtype, dataDim>& x2)
{
    dtype distance = 0.0;
    //#pragma omp simd reduction(+:distance)
    for (std::size_t i = 0; i < dataDim; ++i)
    {
        dtype diff = x1[i] - x2[i];
        distance += diff * diff;
    }

    dtype result = sqrt(distance);

    return result;
}


size_t simple_vote(std::array<std::pair<dtype, size_t>, k>& neighbors)
{
    std::array<size_t, classesNum> votes_to_classes = {};

    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        votes_to_classes[neighbors[i].second]++;
    }

    size_t max_ind = 0;
    size_t max_value = 0;

    for (int i = 0; i < classesNum; ++i)
    {
        if (votes_to_classes[i] > max_value)
        {
            max_value = votes_to_classes[i];
            max_ind = i;
        }
    }

    return max_ind;
}


std::vector<size_t> run_knn(std::vector<std::array<dtype, dataDim>>& train, std::vector<size_t>& train_labels, std::vector<std::array<dtype, dataDim>>& test)
{
    auto train_nrows = train.size();
    std::vector<size_t> predictions(test.size());

    #pragma omp parallel for simd
    for (size_t i = 0; i < test.size(); ++i)
    {
        std::array<std::pair<dtype, size_t>, k> queue_neighbors;

        //count distances
        for (int j = 0; j < k; ++j)
        {
            dtype dist = euclidean_dist(train[j], test[i]);
            queue_neighbors[j] = std::make_pair(dist, train_labels[j]);
        }

        sort_queue(queue_neighbors);

        for (int j = k; j < train_nrows; ++j)
        {
            dtype dist = euclidean_dist(train[j], test[i]);
            auto new_neighbor = std::make_pair(dist, train_labels[j]);

            if (new_neighbor.first < queue_neighbors[k-1].first)
            {
                queue_neighbors[k-1] = new_neighbor;
                push_queue(queue_neighbors, new_neighbor);
            }

        }
        predictions[i] = simple_vote(queue_neighbors);
    }

    return predictions;
}
