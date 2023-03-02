# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import sqrt


def _groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    """Determines the euclidean distance from the cluster center to each point

    Args:
        arrayP (_type_): _description_
        arrayPcluster (_type_): _description_
        arrayC (_type_): _description_
        num_points (_type_): _description_
        num_centroids (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i0 in range(num_points):
        minor_distance = -1
        for i1 in range(num_centroids):
            dx = arrayP[i0, 0] - arrayC[i1, 0]
            dy = arrayP[i0, 1] - arrayC[i1, 1]
            my_distance = sqrt(dx * dx + dy * dy)
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[i0] = i1
    return arrayPcluster


def _calCentroidsSum(
    arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
):
    """Assigns points to cluster

    Args:
        arrayP (_type_): _description_
        arrayPcluster (_type_): _description_
        arrayCsum (_type_): _description_
        arrayCnumpoint (_type_): _description_
        num_points (_type_): _description_
        num_centroids (_type_): _description_

    Returns:
        _type_: _description_
    """
    # parallel for loop
    for i in range(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0

    for i in range(num_points):
        ci = arrayPcluster[i]
        arrayCsum[ci, 0] += arrayP[i, 0]
        arrayCsum[ci, 1] += arrayP[i, 1]
        arrayCnumpoint[ci] += 1

    return arrayCsum, arrayCnumpoint


def _updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    """Update the centroid array after computation

    Args:
        arrayC (_type_): _description_
        arrayCsum (_type_): _description_
        arrayCnumpoint (_type_): _description_
        num_centroids (_type_): _description_
    """
    for i in range(num_centroids):
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


def _kmeans_impl(
    arrayP,
    arrayPcluster,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    num_iters,
    num_points,
    num_centroids,
):
    for i in range(num_iters):
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
    for i1 in range(ncentroids):
        arrayC[i1, 0] = arrayP[i1, 0]
        arrayC[i1, 1] = arrayP[i1, 1]

    arrayC, arrayCsum, arrayCnumpoint = _kmeans_impl(
        arrayP,
        arrayPclusters,
        arrayC,
        arrayCsum,
        arrayCnumpoint,
        niters,
        npoints,
        ncentroids,
    )
