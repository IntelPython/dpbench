# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def kernel(path: dc.int32[N, N]):
    # def kernel(path: dc.float64[N, N]):

    for k in range(N):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
        # for i in range(N):
        #     path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])
