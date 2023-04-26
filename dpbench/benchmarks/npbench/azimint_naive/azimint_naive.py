# SPDX-FileCopyrightText: 2014 Jérôme Kieffer et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    data, radius = rng.random((N,)), rng.random((N,))
    return data, radius
