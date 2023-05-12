# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(nevts, nout, types_dict):
    import numpy as np

    dtype = types_dict["float"]

    C1 = np.empty((nevts, nout), dtype=dtype)
    F1 = np.empty((nevts, nout), dtype=dtype)
    Q1 = np.empty((nevts, nout), dtype=dtype)

    np.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = np.random.rand()
            F1[i, j] = np.random.rand()
            Q1[i, j] = np.random.rand() * np.random.rand()

    return (C1, F1, Q1, np.empty((nevts, nout, 4), dtype))
