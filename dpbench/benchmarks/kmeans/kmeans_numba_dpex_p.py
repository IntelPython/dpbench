# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np
import numba as nb
import numba_dpex as dpex


# determine the euclidean distance from the cluster center to each point
@dpex.dpjit
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    # parallel for loop
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


# assign points to cluster
@dpex.dpjit
def calCentroidsSum(
    arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
):
    # parallel for loop
    for i in nb.prange(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0


@dpex.kernel
def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint):
    i = dpex.get_global_id(0)
    ci = arrayPcluster[i]
    dpex.atomic.add(arrayCsum, (ci, 0), arrayP[i, 0])
    dpex.atomic.add(arrayCsum, (ci, 1), arrayP[i, 1])
    dpex.atomic.add(arrayCnumpoint, ci, 1)


# update the centriods array after computation
@dpex.dpjit
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    for i in nb.prange(num_centroids):
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


@dpex.dpjit
def copy_arrayC(arrayC, arrayP, num_centroids):
    for i in nb.prange(num_centroids):
        arrayC[i, 0] = arrayP[i, 0]
        arrayC[i, 1] = arrayP[i, 1]


def kmeans_numba(
    arrayP,
    arrayPcluster,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    niters,
    num_points,
    num_centroids,
):
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

        calCentroidsSum2[num_points,](
            arrayP, arrayPcluster, arrayCsum, arrayCnumpoint
        )

        updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids)

    return arrayC, arrayCsum, arrayCnumpoint


def kmeans(
    arrayP,
    arrayPclusters,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    niters,
    npoints,
    ndims,
    ncentroids,
):
    copy_arrayC(arrayC, arrayP, ncentroids)

    arrayC, arrayCsum, arrayCnumpoint = kmeans_numba(
        arrayP,
        arrayPclusters,
        arrayC,
        arrayCsum,
        arrayCnumpoint,
        niters,
        npoints,
        ncentroids,
    )
