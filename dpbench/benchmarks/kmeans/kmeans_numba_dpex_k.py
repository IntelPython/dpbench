# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0
import numba_dpex as nb
import numpy
from numba_dpex import atomic

REPEAT = 1

atomic_add = atomic.add


@nb.kernel
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    idx = nb.get_global_id(0)
    # if idx < num_points: # why it was removed??
    minor_distance = -1
    for i in range(num_centroids):
        dx = arrayP[idx, 0] - arrayC[i, 0]
        dy = arrayP[idx, 1] - arrayC[i, 1]
        my_distance = numpy.sqrt(dx * dx + dy * dy)
        if minor_distance > my_distance or minor_distance == -1:
            minor_distance = my_distance
            arrayPcluster[idx] = i


@nb.kernel
def calCentroidsSum1(arrayCsum, arrayCnumpoint):
    i = nb.get_global_id(0)
    arrayCsum[i, 0] = 0
    arrayCsum[i, 1] = 0
    arrayCnumpoint[i] = 0


@nb.kernel
def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint):
    i = nb.get_global_id(0)
    ci = arrayPcluster[i]
    atomic_add(arrayCsum, (ci, 0), arrayP[i, 0])
    atomic_add(arrayCsum, (ci, 1), arrayP[i, 1])
    atomic_add(arrayCnumpoint, ci, 1)


@nb.kernel
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    i = nb.get_global_id(0)
    arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
    arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


@nb.kernel
def copy_arrayC(arrayC, arrayP):
    i = nb.get_global_id(0)
    arrayC[i, 0] = arrayP[i, 0]
    arrayC[i, 1] = arrayP[i, 1]


def kmeans_kernel(
    arrayP,
    arrayPcluster,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    niters,
    num_points,
    num_centroids,
):

    copy_arrayC[num_centroids, nb.DEFAULT_LOCAL_SIZE](arrayC, arrayP)

    for i in range(niters):
        groupByCluster[num_points, nb.DEFAULT_LOCAL_SIZE](
            arrayP, arrayPcluster, arrayC, num_points, num_centroids
        )

        calCentroidsSum1[num_centroids, nb.DEFAULT_LOCAL_SIZE](
            arrayCsum, arrayCnumpoint
        )

        calCentroidsSum2[num_points, nb.DEFAULT_LOCAL_SIZE](
            arrayP, arrayPcluster, arrayCsum, arrayCnumpoint
        )

        updateCentroids[num_centroids, nb.DEFAULT_LOCAL_SIZE](
            arrayC, arrayCsum, arrayCnumpoint, num_centroids
        )

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

    for i in range(REPEAT):
        arrayC, arrayCsum, arrayCnumpoint = kmeans_kernel(
            arrayP,
            arrayPclusters,
            arrayC,
            arrayCsum,
            arrayCnumpoint,
            niters,
            npoints,
            ncentroids,
        )
