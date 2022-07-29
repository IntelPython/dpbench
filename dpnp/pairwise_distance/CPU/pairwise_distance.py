# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base_pair_wise
import dpctl

import dpnp as np


def pw_distance(X1, X2, D):
    # return np.sqrt((np.square(X1 - X2.reshape((X2.shape[0],1,X2.shape[1])))).sum(axis=2))
    x1 = np.sum(np.square(X1), axis=1)  # X1=4*3 -> 4*1
    x2 = np.sum(np.square(X2), axis=1)  # X2=4*3 -> 4*1
    D = -2 * np.dot(X1, X2.T)
    x3 = x1.reshape(x1.size, 1)
    D += x3  # x1[:,None] Not supported by Numba
    D += x2
    D = np.sqrt(D)


def pw_distance_dpctl(X1, X2, D):
    with dpctl.device_context("opencl:cpu"):
        pw_distance(X1, X2, D)


base_pair_wise.run("Numba FastMath", pw_distance_dpctl)
