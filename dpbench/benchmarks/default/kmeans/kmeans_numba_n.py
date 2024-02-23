# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=False, fastmath=True)
def _groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    for i0 in nb.prange(num_points):
        minor_distance = -1
        for i1 in range(num_centroids):
            dx = arrayP[i0, 0] - arrayC[i1, 0]
            dy = arrayP[i0, 1] - arrayC[i1, 1]
            my_distance = np.sqrt(dx * dx + dy * dy)
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[i0] = i1
    return arrayPcluster


@nb.njit(parallel=False, fastmath=True)
def _calCentroidsSum(
    arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
):
    for i in nb.prange(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0

    for i in range(num_points):
        ci = arrayPcluster[i]
        arrayCsum[ci, 0] += arrayP[i, 0]
        arrayCsum[ci, 1] += arrayP[i, 1]
        arrayCnumpoint[ci] += 1

    return arrayCsum, arrayCnumpoint


@nb.njit(parallel=False, fastmath=True)
def _updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    for i in nb.prange(num_centroids):
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


@nb.njit(parallel=False, fastmath=True)
def _kmeans_impl(arrayP, arrayPcluster, arrayC, arrayCnumpoint, niters):
    num_points = arrayP.shape[0]
    num_centroids = arrayC.shape[0]
    arrayCsum = np.zeros_like(arrayC)
    for i in range(niters):
        _groupByCluster(
            arrayP, arrayPcluster, arrayC, num_points, num_centroids
        )

        _calCentroidsSum(
            arrayP,
            arrayPcluster,
            arrayCsum,
            arrayCnumpoint,
            num_points,
            num_centroids,
        )

        _updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids)

    return arrayC, arrayCsum, arrayCnumpoint


@nb.njit(parallel=False, fastmath=True)
def kmeans(
    arrayP,
    arrayPclusters,
    arrayC,
    arrayCnumpoint,
    niters,
):
    arrayC, arrayCsum, arrayCnumpoint = _kmeans_impl(
        arrayP, arrayPclusters, arrayC, arrayCnumpoint, niters
    )
