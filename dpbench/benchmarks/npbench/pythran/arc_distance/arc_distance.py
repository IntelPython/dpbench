# SPDX-FileCopyrightText: 2019 Serge Guelton
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    t0, p0, t1, p1 = (
        rng.random((N,)),
        rng.random((N,)),
        rng.random((N,)),
        rng.random((N,)),
    )
    return t0, p0, t1, p1
