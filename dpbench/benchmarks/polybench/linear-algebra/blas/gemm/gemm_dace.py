# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ("NI", "NJ", "NK"))


@dc.program
def kernel(
    alpha: dc.float64,
    beta: dc.float64,
    C: dc.float64[NI, NJ],
    A: dc.float64[NI, NK],
    B: dc.float64[NK, NJ],
):
    C[:] = alpha * A @ B + beta * C
