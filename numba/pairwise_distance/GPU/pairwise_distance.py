# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_pair_wise
import numpy as np
import numba
import os
from device_selector import get_device_selector

backend = os.getenv("NUMBA_BACKEND", "legacy")
if backend == "legacy":
    import numba as nb

    # Naieve pairwise distance impl - take an array representing M points in N dimensions, and return the M x M matrix of Euclidean distances
    @nb.njit(parallel=True, fastmath=True)
    def pw_distance_kernel(X1, X2, D):
        # Size of imputs
        M = X1.shape[0]
        N = X2.shape[0]
        O = X1.shape[1]

        # Outermost parallel loop over the matrix X1
        for i in numba.prange(M):
            # Loop over the matrix X2
            for j in range(N):
                d = 0.0
                # Compute exclidean distance
                for k in range(O):
                    tmp = X1[i, k] - X2[j, k]
                    d += tmp * tmp
                # Write computed distance to distance matrix
                D[i, j] = np.sqrt(d)


else:
    import numba_dpcomp as nb

    # Naieve pairwise distance impl - take an array representing M points in N dimensions, and return the M x M matrix of Euclidean distances
    @nb.njit(parallel=True, fastmath=True, enable_gpu_pipeline=True)
    def pw_distance_kernel(X1, X2, D):
        # Size of imputs
        M = X1.shape[0]
        N = X2.shape[0]
        O = X1.shape[1]

        # Outermost parallel loop over the matrix X1
        for i in numba.prange(M):
            # Loop over the matrix X2
            for j in range(N):
                d = 0.0
                # Compute exclidean distance
                for k in range(O):
                    tmp = X1[i, k] - X2[j, k]
                    d += tmp * tmp
                # Write computed distance to distance matrix
                D[i, j] = np.sqrt(d)


def pw_distance(X1, X2, D):
    with dpctl.device_context(get_device_selector(is_gpu=True)):
        pw_distance_kernel(X1, X2, D)


base_pair_wise.run("Numba par_for", pw_distance)
