import dpctl
import numpy
import numba
import os

import base_kmeans
from device_selector import get_device_selector

REPEAT = 1
# defines total number of iterations for kmeans accuracy
ITERATIONS = 30

backend = os.getenv("NUMBA_BACKEND", "legacy")
if backend == "legacy":
    import numba as nb

    __njit = nb.jit(nopython=True, parallel=True, fastmath=True)
else:
    import numba_dpcomp as nb

    __njit = nb.jit(nopython=True, parallel=True, fastmath=True, enable_gpu_pipeline=True)


# determine the euclidean distance from the cluster center to each point
@__njit
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    # parallel for loop
    for i0 in numba.prange(num_points):
        minor_distance = -1
        for i1 in range(num_centroids):
            dx = arrayP[i0, 0] - arrayC[i1, 0]
            dy = arrayP[i0, 1] - arrayC[i1, 1]
            my_distance = numpy.sqrt(dx * dx + dy * dy)
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[i0] = i1
    return arrayPcluster


# assign points to cluster
@__njit
def calCentroidsSum(
    arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
):
    # parallel for loop
    for i in numba.prange(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0

    for i in range(num_points):
        ci = arrayPcluster[i]
        arrayCsum[ci, 0] += arrayP[i, 0]
        arrayCsum[ci, 1] += arrayP[i, 1]
        arrayCnumpoint[ci] += 1

    return arrayCsum, arrayCnumpoint


# update the centriods array after computation
@__njit
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    for i in numba.prange(num_centroids):
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


def kmeans(
    arrayP, arrayPcluster, arrayC, arrayCsum, arrayCnumpoint, num_points, num_centroids
):

    for i in range(ITERATIONS):
        groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids)

        calCentroidsSum(
            arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
        )

        updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids)

    return arrayC, arrayCsum, arrayCnumpoint


def run_kmeans(
    arrayP,
    arrayPclusters,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    NUMBER_OF_POINTS,
    NUMBER_OF_CENTROIDS,
):

    with dpctl.device_context(get_device_selector(is_gpu=True)):
        for i in range(REPEAT):
            for i1 in range(NUMBER_OF_CENTROIDS):
                arrayC[i1, 0] = arrayP[i1, 0]
                arrayC[i1, 1] = arrayP[i1, 1]

            arrayC, arrayCsum, arrayCnumpoint = kmeans(
                arrayP,
                arrayPclusters,
                arrayC,
                arrayCsum,
                arrayCnumpoint,
                NUMBER_OF_POINTS,
                NUMBER_OF_CENTROIDS,
            )


base_kmeans.run("Kmeans Numba", run_kmeans)
