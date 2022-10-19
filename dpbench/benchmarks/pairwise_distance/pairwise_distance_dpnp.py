# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

def pairwise_distance(X1, X2, D):
    """Pairwise Numpy implementation using numpy linalg norm"""
    dpnp.copyto(D, dpnp.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1))
