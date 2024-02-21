# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from math import sqrt

import numba_mlir.kernel as nb
import numpy
from dpctl import tensor as dpt
from numba_mlir.kernel import group, local, private

atomic_add = nb.atomic.add
local_array = local.array
private_array = private.array
group_reduce_max = group.reduce_max


def DivUp(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def Align(value, base):
    return base * DivUp(value, base)


@lru_cache(maxsize=1)
def getGroupByCluster(  # noqa: C901
    dims, num_centroids, dtyp, WorkPI, local_size_
):
    local_copies = min(4, max(1, DivUp(local_size_, num_centroids)))

    @nb.kernel(gpu_fp64_truncate="auto")
    def groupByCluster(
        arrayP, arrayPcluster, arrayC, NewCentroids, NewCount, last
    ):
        numpoints = arrayP.shape[0]
        localCentroids = local_array((dims, num_centroids), dtype=dtyp)
        localNewCentroids = local_array(
            (local_copies, dims, num_centroids), dtype=dtyp
        )
        localNewCount = local_array(
            (local_copies, num_centroids), dtype=numpy.int32
        )

        grid = nb.get_group_id(0)
        lid = nb.get_local_id(0)
        local_size = nb.get_local_size(0)

        for i in range(lid, num_centroids * dims, local_size):
            localCentroids[i % dims, i // dims] = arrayC[i // dims, i % dims]
            for lc in range(local_copies):
                localNewCentroids[lc, i % dims, i // dims] = 0

        for c in range(lid, num_centroids, local_size):
            for lc in range(local_copies):
                localNewCount[lc, c] = 0

        nb.barrier(nb.LOCAL_MEM_FENCE)

        for i in range(WorkPI):
            point_id = grid * WorkPI * local_size + i * local_size + lid
            if point_id < numpoints:
                localP = private_array(dims, dtyp)
                for d in range(dims):
                    localP[d] = arrayP[point_id, d]

                minimal_distance = dtyp.type(dpt.inf)
                nearest_centroid = int(0)
                for c in range(num_centroids):
                    sq_sum = dtyp.type(0)
                    for d in range(dims):
                        sq_sum += (localP[d] - localCentroids[d, c]) ** 2

                    if sq_sum < minimal_distance:
                        nearest_centroid = c
                        minimal_distance = sq_sum

                lc = lid % local_copies
                for d in range(dims):
                    atomic_add(
                        localNewCentroids, (lc, d, nearest_centroid), localP[d]
                    )

                atomic_add(localNewCount, (lc, nearest_centroid), 1)

                if last:
                    arrayPcluster[point_id] = nearest_centroid

        nb.barrier(nb.LOCAL_MEM_FENCE)

        for i in range(lid, num_centroids * dims, local_size):
            local_centroid_d = dtyp.type(0)
            for lc in range(local_copies):
                local_centroid_d += localNewCentroids[lc, i % dims, i // dims]

            atomic_add(
                NewCentroids,
                (i // dims, i % dims),
                local_centroid_d,
            )

        for c in range(lid, num_centroids, local_size):
            local_centroid_npoints = numpy.int32(0)
            for lc in range(local_copies):
                local_centroid_npoints += localNewCount[lc, c]

            atomic_add(NewCount, c, local_centroid_npoints)

    return groupByCluster


@lru_cache(maxsize=1)
def getUpdateCentroids(dims, num_centroids, dtyp, local_size_):
    @nb.kernel(gpu_fp64_truncate="auto")
    def updateCentroids(diff, arrayC, arrayCnumpoint, NewCentroids, NewCount):
        lid = nb.get_local_id(0)
        local_size = nb.get_local_size(0)

        local_distance = local_array(local_size_, dtype=dtyp)

        max_distance = dtyp.type(0)
        for c in range(lid, num_centroids, local_size):
            numpoints = NewCount[c]
            arrayCnumpoint[c] = numpoints
            NewCount[c] = 0
            distance = dtyp.type(0)

            for d in range(dims):
                d0 = arrayC[c, d]
                d1 = NewCentroids[c, d]
                NewCentroids[c, d] = 0

                d1 = d1 / numpoints if numpoints > 0 else d0
                arrayC[c, d] = d1

                distance += d0 * d0 - d1 * d1

            max_distance = max(max_distance, distance)
            local_distance[c] = max_distance

        nb.barrier(nb.LOCAL_MEM_FENCE)

        if lid == 0:
            max_distance = group_reduce_max(local_distance[c])

            diff[0] = sqrt(max_distance)

    return updateCentroids


@lru_cache(maxsize=1)
def getUpdateLabels(dims, num_centroids, dtyp, WorkPI):
    @nb.kernel(gpu_fp64_truncate="auto")
    def updateLabels(arrayP, arrayPcluster, arrayC):
        numpoints = arrayP.shape[0]
        localCentroids = local_array((dims, num_centroids), dtype=dtyp)

        grid = nb.get_group_id(0)
        lid = nb.get_local_id(0)
        local_size = nb.get_local_size(0)

        for i in range(lid, num_centroids * dims, local_size):
            localCentroids[i % dims, i // dims] = arrayC[i // dims, i % dims]

        nb.barrier(nb.LOCAL_MEM_FENCE)

        for i in range(WorkPI):
            point_id = grid * WorkPI * local_size + i * local_size + lid
            if point_id < numpoints:
                localP = private_array(dims, dtyp)
                for d in range(dims):
                    localP[d] = arrayP[point_id, d]

                minimal_distance = dtyp.type(dpt.inf)
                nearest_centroid = int(0)
                for c in range(num_centroids):
                    sq_sum = dtyp.type(0)
                    for d in range(dims):
                        sq_sum += (localP[d] - localCentroids[d, c]) ** 2

                    if sq_sum < minimal_distance:
                        nearest_centroid = c
                        minimal_distance = sq_sum

                arrayPcluster[point_id] = nearest_centroid

    return updateLabels


@lru_cache(maxsize=1)
def getKernels(dims, num_centroids, dtyp, WorkPI, local_size_):
    groupByCluster = getGroupByCluster(
        dims, num_centroids, dtyp, WorkPI, local_size_
    )
    updateCentroids = getUpdateCentroids(dims, num_centroids, dtyp, local_size_)
    updateLabels = getUpdateLabels(dims, num_centroids, dtyp, WorkPI)

    return groupByCluster, updateCentroids, updateLabels


def kmeans_kernel(
    arrayP,
    arrayPcluster,
    arrayC,
    arrayCnumpoint,
    niters,
):
    num_points = arrayP.shape[0]
    num_centroids = arrayC.shape[0]
    dims = arrayC.shape[1]
    device = arrayP.device

    NewCentroids = dpt.zeros_like(arrayC)
    NewCount = dpt.zeros_like(arrayCnumpoint)
    diff = dpt.zeros(1, dtype=arrayP.dtype, device=device)
    diff_host = dpt.inf

    tolerance = 0
    WorkPI = 8
    local_size = min(1024, device.sycl_device.max_work_group_size)
    global_size = Align(DivUp(num_points, WorkPI), local_size)

    groupByCluster, updateCentroids, updateLabels = getKernels(
        dims, num_centroids, arrayP.dtype, WorkPI, local_size
    )

    for i in range(niters):
        last = i == (niters - 1)
        if diff_host < tolerance:
            updateLabels[(global_size,), (local_size,)](
                arrayP, arrayPcluster, arrayC
            )
            break

        groupByCluster[(global_size,), (local_size,)](
            arrayP, arrayPcluster, arrayC, NewCentroids, NewCount, last
        )

        update_centroid_size = min(num_centroids, local_size)
        updateCentroids[(update_centroid_size,), (update_centroid_size,)](
            diff, arrayC, arrayCnumpoint, NewCentroids, NewCount
        )
        diff_host = dpt.asnumpy(diff)[0]


def kmeans(arrayP, arrayPclusters, arrayC, arrayCnumpoint, niters):
    kmeans_kernel(arrayP, arrayPclusters, arrayC, arrayCnumpoint, niters)
