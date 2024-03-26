# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from math import sqrt

import numba_dpex as dpex
from dpctl import tensor as dpt
from numba_dpex import kernel_api as kapi


def DivUp(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def Align(value, base):
    return base * DivUp(value, base)


@lru_cache(maxsize=1)
def getGroupByCluster(  # noqa: C901
    dims, num_centroids, dtyp, WorkPI, local_size_
):
    @dpex.kernel
    def groupByCluster(
        nd_item: kapi.NdItem,
        arrayP,
        arrayPcluster,
        arrayC,
        NewCentroids,
        NewCount,
        last,
        local_copies,
        localCentroids,
        localNewCentroids,
        localNewCount,
    ):
        numpoints = arrayP.shape[0]

        grid = nd_item.get_group().get_group_id(0)
        lid = nd_item.get_local_id(0)
        local_size = nd_item.get_local_range(0)

        for i in range(lid, num_centroids * dims, local_size):
            localCentroids[i % dims, i // dims] = arrayC[i // dims, i % dims]
            for lc in range(local_copies):
                localNewCentroids[lc, i % dims, i // dims] = 0

        for c in range(lid, num_centroids, local_size):
            for lc in range(local_copies):
                localNewCount[lc, c] = 0

        kapi.group_barrier(nd_item.get_group())

        for i in range(WorkPI):
            point_id = grid * WorkPI * local_size + i * local_size + lid
            if point_id < numpoints:
                localP = dpex.private.array(dims, dtyp)
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
                    localNewCentroids_aref = kapi.AtomicRef(
                        localNewCentroids,
                        index=(lc, d, nearest_centroid),
                        address_space=kapi.AddressSpace.LOCAL,
                    )
                    localNewCentroids_aref.fetch_add(localP[d])

                localNewCount_aref = kapi.AtomicRef(
                    localNewCount,
                    index=(lc, nearest_centroid),
                    address_space=kapi.AddressSpace.LOCAL,
                )
                localNewCount_aref.fetch_add(1)

                if last:
                    arrayPcluster[point_id] = nearest_centroid

        kapi.group_barrier(nd_item.get_group())

        for i in range(lid, num_centroids * dims, local_size):
            local_centroid_d = dtyp.type(0)
            for lc in range(local_copies):
                local_centroid_d += localNewCentroids[lc, i % dims, i // dims]

            NewCentroids_aref = kapi.AtomicRef(
                NewCentroids, index=(i // dims, i % dims)
            )
            NewCentroids_aref.fetch_add(local_centroid_d)

        for c in range(lid, num_centroids, local_size):
            local_centroid_npoints = dpt.int32.type(0)
            for lc in range(local_copies):
                local_centroid_npoints += localNewCount[lc, c]

            NewCount_aref = kapi.AtomicRef(NewCount, index=c)
            NewCount_aref.fetch_add(local_centroid_npoints)

    return groupByCluster


@lru_cache(maxsize=1)
def getUpdateCentroids(dims, num_centroids, dtyp, local_size_):
    @dpex.kernel
    def updateCentroids(
        nd_item: kapi.NdItem,
        diff,
        arrayC,
        arrayCnumpoint,
        NewCentroids,
        NewCount,
        local_distance,
    ):
        lid = nd_item.get_local_id(0)
        local_size = nd_item.get_local_range(0)

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

                d1 = d1 / dtyp.type(numpoints) if numpoints > 0 else d0
                arrayC[c, d] = d1

                distance += d0 * d0 - d1 * d1

            max_distance = max(max_distance, distance)
            local_distance[c] = max_distance

        kapi.group_barrier(nd_item.get_group())

        if lid == 0:
            for c in range(local_size):
                max_distance = max(max_distance, local_distance[c])

            diff[0] = sqrt(max_distance)

    return updateCentroids


@lru_cache(maxsize=1)
def getUpdateLabels(dims, num_centroids, dtyp, WorkPI):
    @dpex.kernel
    def updateLabels(
        nd_item: kapi.NdItem, arrayP, arrayPcluster, arrayC, localCentroids
    ):
        numpoints = arrayP.shape[0]

        grid = nd_item.get_group().get_group_id(0)
        lid = nd_item.get_local_id(0)
        local_size = nd_item.get_local_range(0)

        for i in range(lid, num_centroids * dims, local_size):
            localCentroids[i % dims, i // dims] = arrayC[i // dims, i % dims]

        kapi.group_barrier(nd_item.get_group())

        for i in range(WorkPI):
            point_id = grid * WorkPI * local_size + i * local_size + lid
            if point_id < numpoints:
                localP = dpex.private.array(dims, dtyp)
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

    NewCentroids = dpt.zeros_like(arrayC._array_obj)
    NewCount = dpt.zeros_like(arrayCnumpoint._array_obj)
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
            localCentroids = kapi.LocalAccessor(
                (dims, num_centroids), dtype=arrayP.dtype
            )

            dpex.call_kernel(
                updateLabels,
                kapi.NdRange((global_size,), (local_size,)),
                arrayP,
                arrayPcluster,
                arrayC,
                localCentroids,
            )
            break

        local_copies = min(4, max(1, DivUp(local_size, num_centroids)))
        localCentroids = kapi.LocalAccessor(
            (dims, num_centroids), dtype=arrayP.dtype
        )
        localNewCentroids = kapi.LocalAccessor(
            (local_copies, dims, num_centroids), dtype=arrayP.dtype
        )
        localNewCount = kapi.LocalAccessor(
            (local_copies, num_centroids), dtype=dpt.int64
        )
        dpex.call_kernel(
            groupByCluster,
            kapi.NdRange((global_size,), (local_size,)),
            arrayP,
            arrayPcluster,
            arrayC,
            NewCentroids,
            NewCount,
            last,
            local_copies,
            localCentroids,
            localNewCentroids,
            localNewCount,
        )

        local_distance = kapi.LocalAccessor(local_size, dtype=arrayP.dtype)
        update_centroid_size = min(num_centroids, local_size)
        dpex.call_kernel(
            updateCentroids,
            kapi.NdRange((update_centroid_size,), (update_centroid_size,)),
            diff,
            arrayC,
            arrayCnumpoint,
            NewCentroids,
            NewCount,
            local_distance,
        )
        diff_host = dpt.asnumpy(diff)[0]


def kmeans(arrayP, arrayPclusters, arrayC, arrayCnumpoint, niters):
    kmeans_kernel(arrayP, arrayPclusters, arrayC, arrayCnumpoint, niters)
