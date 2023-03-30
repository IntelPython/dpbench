# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp


def _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins):
    dm = (
        dpnp.square(x2 - x1[:, None])
        + dpnp.square(y2 - y1[:, None])
        + dpnp.square(z2 - z1[:, None])
    )
    return dpnp.array(
        [dpnp.outer(w1, w2)[dm <= rbins[k]].sum() for k in range(len(rbins))]
    )


def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
    results[:] = _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins)
