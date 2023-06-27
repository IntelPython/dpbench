# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba
import numba_mlir as nb
import numba_mlir.kernel as nbk
import numpy as np
from dpctl import tensor as dpt

atomic_add = nbk.atomic.add


# determine the euclidean distance from the cluster center to each point
@nb.njit(parallel=True, fastmath=True, gpu_fp64_truncate="auto")
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    # parallel for loop
    for i0 in numba.prange(num_points):
        minor_distance = -1
        for i1 in range(num_centroids):
            dx = arrayP[i0, 0] - arrayC[i1, 0]
            dy = arrayP[i0, 1] - arrayC[i1, 1]
            my_distance = np.sqrt(dx * dx + dy * dy)
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[i0] = i1
    return arrayPcluster


# assign points to cluster
@nb.njit(parallel=True, fastmath=True, gpu_fp64_truncate="auto")
def calCentroidsSum(
    arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
):
    # parallel for loop
    for i in numba.prange(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0


@nbk.kernel(gpu_fp64_truncate="auto")
def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint):
    i = nbk.get_global_id(0)
    ci = arrayPcluster[i]
    atomic_add(arrayCsum, (ci, 0), arrayP[i, 0])
    atomic_add(arrayCsum, (ci, 1), arrayP[i, 1])
    atomic_add(arrayCnumpoint, ci, 1)


# update the centriods array after computation
@nb.njit(parallel=True, fastmath=True, gpu_fp64_truncate="auto")
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    for i in numba.prange(num_centroids):
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


def kmeans_numba(arrayP, arrayPcluster, arrayC, arrayCnumpoint, niters):
    num_points = arrayP.shape[0]
    num_centroids = arrayC.shape[0]
    arrayCsum = dpt.zeros_like(arrayC)

    for i in range(niters):
        groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids)

        calCentroidsSum(
            arrayP,
            arrayPcluster,
            arrayCsum,
            arrayCnumpoint,
            num_points,
            num_centroids,
        )

        calCentroidsSum2[num_points, ()](
            arrayP, arrayPcluster, arrayCsum, arrayCnumpoint
        )

        updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids)

    return arrayC, arrayCsum, arrayCnumpoint


def kmeans(arrayP, arrayPclusters, arrayC, arrayCnumpoint, niters):
    arrayC, arrayCsum, arrayCnumpoint = kmeans_numba(
        arrayP, arrayPclusters, arrayC, arrayCnumpoint, niters
    )
