# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import cupy as cp


def l2_norm(a, d):
    sq = cp.square(a)
    sum = sq.sum(axis=1)
    d[:] = cp.sqrt(sum)

    cp.cuda.stream.get_current_stream().synchronize()
