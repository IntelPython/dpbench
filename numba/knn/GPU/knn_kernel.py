# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import numpy as np
import numba
import base_knn
import dpctl
import numba_dppy

# Define the number of nearest neighbors
k = 5

# @numba.jit(nopython=True)
# def euclidean_dist(x1, x2):
#     return np.linalg.norm(x1-x2)


@numba.jit(nopython=True)
def euclidean_dist(x1, x2):
    distance = 0

    for i in range(base_knn.DATA_DIM):
        diff = x1[i] - x2[i]
        distance += diff * diff

    result = distance ** 0.5
    # result = np.sqrt(distance)
    return result

@numba.jit(nopython=True)
def push_queue(queue_neighbors, new_distance, index=4):
    while (index > 0 and new_distance[0] < queue_neighbors[index - 1,0]):
        queue_neighbors[index] = queue_neighbors[index - 1]
        index = index - 1
        queue_neighbors[index] = new_distance


@numba.jit(nopython=True)
def sort_queue(queue_neighbors):
    for i in range(len(queue_neighbors)):
        push_queue(queue_neighbors, queue_neighbors[i], i)



@numba.jit(nopython=True)
def simple_vote(neighbors, classes_num):
    votes_to_classes = np.zeros(classes_num)

    for i in range(len(neighbors)):
        votes_to_classes[int(neighbors[i,1])] += 1

    max_ind = 0
    max_value = 0

    for i in range(classes_num):
        if (votes_to_classes[i] > max_value):
            max_value = votes_to_classes[i]
            max_ind = i

    return max_ind


@numba_dppy.kernel
def run_knn_kernel(train, train_labels, test, k, classes_num, train_size, predictions, queue_neighbors_lst, votes_to_classes_lst):
    i = numba_dppy.get_global_id(0)
    queue_neighbors = queue_neighbors_lst[i]

    for j in range(k):
        #dist = euclidean_dist(train[j], test[i])
        x1 = train[j]
        x2 = test[i]

        distance = 0.0            
        for jj in range(base_knn.DATA_DIM):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = distance ** 0.5    
            
        # queue_neighbors[j] = (dist, train_labels[j])
        queue_neighbors[j,0] = dist
        queue_neighbors[j,1] = train_labels[j]
        #queue_neighbors.append((dist, train_labels[j]))

    #sort_queue(queue_neighbors)
    for j in range(1,len(queue_neighbors)):
        #push_queue(queue_neighbors, queue_neighbors[i], i)
        new_distance = queue_neighbors[j]
        index = j
        while (index > 0 and new_distance[0] < queue_neighbors[index - 1,0]):
            queue_neighbors[index,0] = queue_neighbors[index - 1,0]
            queue_neighbors[index,1] = queue_neighbors[index - 1,1]
            
            index = index - 1
            
            queue_neighbors[index,0] = new_distance[0]
            queue_neighbors[index,1] = new_distance[1]       

    for j in range(k, train_size):
        #dist = euclidean_dist(train[j], test[i])
        x1 = train[j]
        x2 = test[i]

        distance = 0.0
        for jj in range(base_knn.DATA_DIM):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = distance ** 0.5
            

        if (dist < queue_neighbors[k - 1][0]):
            #queue_neighbors[k - 1] = new_neighbor
            queue_neighbors[k - 1][0] = dist
            queue_neighbors[k - 1][1] = train_labels[j]
            #push_queue(queue_neighbors, queue_neighbors[k - 1])
            new_distance = queue_neighbors[k - 1]
            index = 4                
            while (index > 0 and new_distance[0] < queue_neighbors[index - 1,0]):
                queue_neighbors[index,0] = queue_neighbors[index - 1,0]
                queue_neighbors[index,1] = queue_neighbors[index - 1,1]
                
                index = index - 1
                
                queue_neighbors[index,0] = new_distance[0]
                queue_neighbors[index,1] = new_distance[1]                

    #predictions[i] = simple_vote(queue_neighbors, classes_num)
    votes_to_classes = votes_to_classes_lst[i]#np.zeros(classes_num)

    for j in range(len(queue_neighbors)):
        votes_to_classes[int(queue_neighbors[j,1])] += 1

    max_ind = 0
    max_value = 0

    for j in range(classes_num):
        if (votes_to_classes[j] > max_value):
            max_value = votes_to_classes[j]
            max_ind = j

    predictions[i] =  max_ind

def run_knn(train, train_labels, test, k, classes_num, test_size, train_size, predictions, queue_neighbors_lst, votes_to_classes_lst):
    with dpctl.device_context(base_knn.get_device_selector()):
        run_knn_kernel[test_size,8](train, train_labels, test, k, classes_num, train_size, predictions, queue_neighbors_lst, votes_to_classes_lst)

base_knn.run("K-Nearest-Neighbors Numba", run_knn)
