# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import cupy as np


def _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins):
    dm = (
        np.square(x2 - x1[:, None])
        + np.square(y2 - y1[:, None])
        + np.square(z2 - z1[:, None])
    )
    return np.array(
        [np.outer(w1, w2)[dm <= rbins[k]].sum() for k in range(len(rbins))]
    )


def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
    results[:] = _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins)
