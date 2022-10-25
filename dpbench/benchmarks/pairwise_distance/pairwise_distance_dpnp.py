# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

def pairwise_distance(X1, X2, D):
    x1 = dpnp.sum(dpnp.square(X1), axis=1)
    x2 = dpnp.sum(dpnp.square(X2), axis=1)
    D = -2 * dpnp.dot(X1, X2.T)
    x3 = x1.reshape(x1.size, 1)
    D += x3  # x1[:,None] Not supported by Numba
    D += x2
    D = dpnp.sqrt(D)
