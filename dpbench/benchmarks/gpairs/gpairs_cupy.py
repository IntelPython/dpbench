# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import cupy as cp


def _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins):
    dm = (
        cp.square(x2 - x1[:, None])
        + cp.square(y2 - y1[:, None])
        + cp.square(z2 - z1[:, None])
    )
    ret_arr = cp.array(
        [cp.outer(w1, w2)[dm <= rbins[k]].sum() for k in range(len(rbins))]
    )

    cp.cuda.stream.get_current_stream().synchronize()

    return ret_arr


def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
    results[:] = _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins)
