# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

def pairwise_distance(X1, X2, D):
    x1 = dpnp.sum(dpnp.square(X1), axis=1)
    x2 = dpnp.sum(dpnp.square(X2), axis=1)
    dpnp.copyto(D, dpnp.dot(X1, X2.T))
    dpnp.copyto(D, dpnp.multiply(D, -2))
    x3 = x1.reshape(x1.size, 1)
    dpnp.copyto(D, dpnp.add(D, x3))
    dpnp.copyto(D, dpnp.add(D, x2))
    dpnp.copyto(D, dpnp.sqrt(D))