# SPDX-FileCopyrightText: 2014-2021 ETH Zurich
# SPDX-FileCopyrightText: 2007 Free Software Foundation, Inc. <https://fsf.org/>
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


def initialize(I, J, K):  # noqa: E741 math variable
    from numpy.random import default_rng

    rng = default_rng(42)

    # Define arrays
    in_field = rng.random((I + 4, J + 4, K))
    out_field = rng.random((I, J, K))
    coeff = rng.random((I, J, K))

    return in_field, out_field, coeff
