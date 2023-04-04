# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import numba_dpex as dpex
import numpy as np

# This implementation is numba dpex kernel version with atomics.


@dpex.kernel
def count_weighted_pairs_3d_intel_no_slm_ker(
    n,
    nbins,
    slm_hist_size,
    private_hist_size,
    x0,
    y0,
    z0,
    w0,
    x1,
    y1,
    z1,
    w1,
    rbins_squared,
    result,
):
    lid0 = dpex.get_local_id(0)
    gr0 = dpex.get_group_id(0)

    lid1 = dpex.get_local_id(1)
    gr1 = dpex.get_group_id(1)

    lws0 = dpex.get_local_size(0)
    lws1 = dpex.get_local_size(1)

    n_wi = 20

    dsq_mat = dpex.private.array(shape=(20 * 20), dtype=np.float32)
    w0_vec = dpex.private.array(shape=(20), dtype=np.float32)
    w1_vec = dpex.private.array(shape=(20), dtype=np.float32)

    offset0 = gr0 * n_wi * lws0 + lid0
    offset1 = gr1 * n_wi * lws1 + lid1

    # work item works on pointer
    # j0 = gr0 * n_wi * lws0 + i0 * lws0 + lid0, and
    # j1 = gr1 * n_wi * lws1 + i1 * lws1 + lid1

    j1 = offset1
    i1 = 0
    while (i1 < n_wi) and (j1 < n):
        w1_vec[i1] = w1[j1]
        i1 += 1
        j1 += lws1

    # compute (n_wi, n_wi) matrix of squared distances in work-item
    j0 = offset0
    i0 = 0
    while (i0 < n_wi) and (j0 < n):
        x0v = x0[j0]
        y0v = y0[j0]
        z0v = z0[j0]
        w0_vec[i0] = w0[j0]

        j1 = offset1
        i1 = 0
        while (i1 < n_wi) and (j1 < n):
            dx = x0v - x1[j1]
            dy = y0v - y1[j1]
            dz = z0v - z1[j1]
            dsq_mat[i0 * n_wi + i1] = dx * dx + dy * dy + dz * dz
            i1 += 1
            j1 += lws1

        i0 += 1
        j0 += lws0

    # update slm_hist. Use work-item private buffer of 16 tfloat elements
    for k in range(0, slm_hist_size, private_hist_size):
        private_hist = dpex.private.array(shape=(16), dtype=np.float32)
        for p in range(private_hist_size):
            private_hist[p] = 0.0

        j0 = offset0
        i0 = 0
        while (i0 < n_wi) and (j0 < n):
            j1 = offset1
            i1 = 0
            while (i1 < n_wi) and (j1 < n):
                dsq = dsq_mat[i0 * n_wi + i1]
                pw = w0_vec[i0] * w1_vec[i1]
                # i1 += 1
                # j1 += lws1
                pk = k
                for p in range(private_hist_size):
                    private_hist[p] += (
                        pw if (pk < nbins and dsq <= rbins_squared[pk]) else 0.0
                    )
                    pk += 1

                i1 += 1
                j1 += lws1

            i0 += 1
            j0 += lws0

        pk = k
        for p in range(private_hist_size):
            dpex.atomic.add(result, pk, private_hist[p])
            pk += 1


def ceiling_quotient(n, m):
    return int((n + m - 1) / m)


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
    n_wi = 20
    private_hist_size = 16
    lws0 = 16
    lws1 = 16

    m0 = n_wi * lws0
    m1 = n_wi * lws1

    n_groups0 = ceiling_quotient(nopt, m0)
    n_groups1 = ceiling_quotient(nopt, m1)

    gwsRange = n_groups0 * lws0, n_groups1 * lws1
    lwsRange = lws0, lws1

    slm_hist_size = (
        ceiling_quotient(nbins, private_hist_size) * private_hist_size
    )

    count_weighted_pairs_3d_intel_no_slm_ker[gwsRange, lwsRange](
        nopt,
        nbins,
        slm_hist_size,
        private_hist_size,
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
    )
