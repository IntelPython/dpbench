import dpctl
import base_kmeans
import numpy
# import numba_dppy
from numba_dpcomp.mlir.kernel_impl import kernel, get_global_id, atomic, DEFAULT_LOCAL_SIZE
atomic_add = atomic.add

REPEAT = 1

ITERATIONS = 30


@kernel
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    idx = get_global_id(0)
    minor_distance = -1
    for i in range(num_centroids):
        dx = arrayP[idx, 0] - arrayC[i, 0]
        dy = arrayP[idx, 1] - arrayC[i, 1]
        my_distance = numpy.sqrt(dx * dx + dy * dy)
        if minor_distance > my_distance or minor_distance == -1:
            minor_distance = my_distance
            arrayPcluster[idx] = i


@kernel
def calCentroidsSum1(arrayCsum, arrayCnumpoint):
    i = get_global_id(0)
    arrayCsum[i, 0] = 0
    arrayCsum[i, 1] = 0
    arrayCnumpoint[i] = 0


@kernel
def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint):
    i = get_global_id(0)
    ci = arrayPcluster[i]
    atomic_add(arrayCsum, (ci, 0), arrayP[i, 0])
    atomic_add(arrayCsum, (ci, 1), arrayP[i, 1])
    atomic_add(arrayCnumpoint, ci, 1)


@kernel
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    i = get_global_id(0)
    arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
    arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


@kernel
def copy_arrayC(arrayC, arrayP):
    i = get_global_id(0)
    arrayC[i, 0] = arrayP[i, 0]
    arrayC[i, 1] = arrayP[i, 1]


def kmeans(
    arrayP, arrayPcluster, arrayC, arrayCsum, arrayCnumpoint, num_points, num_centroids
):

    copy_arrayC[num_centroids, DEFAULT_LOCAL_SIZE](arrayC, arrayP)

    for i in range(ITERATIONS):
        groupByCluster[num_points, DEFAULT_LOCAL_SIZE](
            arrayP, arrayPcluster, arrayC, num_points, num_centroids
        )

        calCentroidsSum1[num_centroids, DEFAULT_LOCAL_SIZE](
            arrayCsum,
            arrayCnumpoint,
        )

        calCentroidsSum2[num_points, DEFAULT_LOCAL_SIZE](
            arrayP,
            arrayPcluster,
            arrayCsum,
            arrayCnumpoint,
        )

        updateCentroids[num_centroids, DEFAULT_LOCAL_SIZE](
            arrayC, arrayCsum, arrayCnumpoint, num_centroids
        )

    return arrayC, arrayCsum, arrayCnumpoint


def printCentroid(arrayC, arrayCsum, arrayCnumpoint, NUMBER_OF_CENTROIDS):
    for i in range(NUMBER_OF_CENTROIDS):
        print(
            "[x={:6f}, y={:6f}, x_sum={:6f}, y_sum={:6f}, num_points={:d}]".format(
                arrayC[i, 0],
                arrayC[i, 1],
                arrayCsum[i, 0],
                arrayCsum[i, 1],
                arrayCnumpoint[i],
            )
        )

    print("--------------------------------------------------")


def run_kmeans(
    arrayP,
    arrayPclusters,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    NUMBER_OF_POINTS,
    NUMBER_OF_CENTROIDS,
):

    with dpctl.device_context(base_kmeans.get_device_selector()):
        for i in range(REPEAT):
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
