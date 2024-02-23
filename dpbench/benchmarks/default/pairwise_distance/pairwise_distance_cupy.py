# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import cupy as cp


def pairwise_distance(X1, X2, D):
    x1 = cp.sum(cp.square(X1), axis=1)
    x2 = cp.sum(cp.square(X2), axis=1)
    cp.dot(X1, X2.T, D)
    D *= -2
    x3 = x1.reshape(x1.size, 1)
    cp.add(D, x3, D)
    cp.add(D, x2, D)
    cp.sqrt(D, D)

    cp.cuda.stream.get_current_stream().synchronize()
