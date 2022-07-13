# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

def pairwise_distance(X1, X2, D):
    """Pairwise Numpy implementation using numpy linalg norm
    """
    np.copyto(D, np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1))
