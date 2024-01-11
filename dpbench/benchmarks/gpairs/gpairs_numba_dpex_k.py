# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache

import numba_dpex as dpex
import numpy
from dpctl import tensor as dpt
from numba_dpex import NdRange


def DivUp(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def Align(value, base):
    return base * DivUp(value, base)


@dpex.func
def upper_bound(arr, size, value):
    first = 0
    count = size

    while count > 0:
        it = first
        step = count // 2
        it += step

        if value >= arr[it]:
            first = it + 1
            count -= step + 1
        else:
            count = step

    return first


@lru_cache(maxsize=1)
def getKernel(  # noqa: C901
    _nbins,
    WPI,
    local_copies,
    dtyp,
    work_group_size,
):
    @dpex.kernel
    def gpairs_impl(n, nbins, x0, y0, z0, w0, x1, y1, z1, w1, rbins, hist):
        gid0_ = WPI * dpex.get_global_id(1)
        gid1 = dpex.get_global_id(0)

        lid0 = dpex.get_local_id(1)
        lid1 = dpex.get_local_id(0)

        lra = dpex.get_local_size(0)

        lid = lid1 + lid0 * lra

        localBins = dpex.local.array((_nbins), dtype=dtyp)
        for i in range(lid, nbins, work_group_size):
            localBins[i] = rbins[i]

        localHist = dpex.local.array((local_copies, _nbins), dtype=dtyp)
        for i in range(lid, nbins, work_group_size):
            for j in range(local_copies):
                localHist[j, i] = dtyp.type(0)

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        _x0 = x0[gid1]
        _y0 = y0[gid1]
        _z0 = z0[gid1]
        _w0 = w0[gid1]

        lastBin = localBins[nbins - 1]

        for i in range(WPI):
            gid0 = gid0_ + i
            if gid0 < n and gid1 < n:
                _x1 = x1[gid0]
                _y1 = y1[gid0]
                _z1 = z1[gid0]
                _w1 = w1[gid0]

                dist = (_x0 - _x1) ** 2 + (_y0 - _y1) ** 2 + (_z0 - _z1) ** 2
                if dist < lastBin:
                    bin_id = upper_bound(localBins, nbins, dist)

                    dpex.atomic.add(
                        localHist, (lid % local_copies, bin_id), _w0 * _w1
                    )

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        for i in range(lid, nbins, work_group_size):
            hist_val = dtyp.type(0)
            for j in range(local_copies):
                hist_val += localHist[j, i]

            dpex.atomic.add(hist, i, hist_val)

    return gpairs_impl


@dpex.kernel
def prefix_sum_impl(hist, hist_size):
    part_sum = hist[0]
    for i in range(1, hist_size):
        part_sum += hist[i]
        hist[i] = part_sum


def gpairs(
    nopt,
    nbins,
    x1,
    y1,
    z1,
    w1,
    x2,
    y2,
    z2,
    w2,
    rbins,
    results,
):
    local_copies = 16
    WPI = 64
    max_worg_group_size = 256
    dim1 = 64

    local_size = dim1, max_worg_group_size // dim1
    work_size = Align(nopt, local_size[0]), Align(
        DivUp(nopt, WPI), local_size[1]
    )

    kernel = getKernel(nbins, WPI, local_copies, x1.dtype, max_worg_group_size)

    kernel[NdRange(work_size, local_size)](
        nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results
    )

    prefix_sum_impl[NdRange((1,), (1,))](results, nbins)
