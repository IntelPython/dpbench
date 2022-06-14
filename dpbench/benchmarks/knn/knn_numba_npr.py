# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import numba

@numba.njit(parallel=True)
def knn(
    x_train,
    y_train,
    x_test,
    k,
    classes_num,
    test_size,
    train_size,
    predictions,
    votes_to_classes,
    data_dim
):

    for i in numba.prange(test_size):
        queue_neighbors = np.empty(shape=(k,2))

        for j in range(k):
            x1 = x_train[j]
            x2 = x_test[i]

            distance = 0.0
            for jj in range(data_dim):
                diff = x1[jj] - x2[jj]
                distance += diff * diff
            dist = math.sqrt(distance)

            queue_neighbors[j, 0] = dist
            queue_neighbors[j, 1] = y_train[j]

        # sort_queue(queue_neighbors)
        for j in range(k):
            new_distance = queue_neighbors[j, 0]
            new_neighbor_label = queue_neighbors[j, 1]
            index = j

            while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
                queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
                queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

                index = index - 1

                queue_neighbors[index, 0] = new_distance
                queue_neighbors[index, 1] = new_neighbor_label

        for j in range(k, train_size):
            # dist = euclidean_dist(x_train[j], x_test[i])
            x1 = x_train[j]
            x2 = x_test[i]

            distance = 0.0
            for jj in range(data_dim):
                diff = x1[jj] - x2[jj]
                distance += diff * diff
            dist = math.sqrt(distance)

            if dist < queue_neighbors[k - 1][0]:
                queue_neighbors[k - 1][0] = dist
                queue_neighbors[k - 1][1] = y_train[j]
                # push_queue(queue_neighbors, queue_neighbors[k - 1])
                new_distance = queue_neighbors[k - 1, 0]
                new_neighbor_label = queue_neighbors[k - 1, 1]
                index = k - 1

                while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
                    queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
                    queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

                    index = index - 1
                    
                    queue_neighbors[index, 0] = new_distance
                    queue_neighbors[index, 1] = new_neighbor_label

        v_to_c_i = votes_to_classes[i]

        for j in range(k):
            v_to_c_i[int(queue_neighbors[j, 1])] += 1

        max_ind = 0
        max_value = 0

        for j in range(classes_num):
            if v_to_c_i[j] > max_value:
                max_value = v_to_c_i[j]
                max_ind = j

        predictions[i] = max_ind
