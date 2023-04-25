# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

NI, NJ, NK, NL, NM = (
    dc.symbol(s, dtype=dc.int64) for s in ("NI", "NJ", "NK", "NL", "NM")
)


@dc.program
def kernel(
    A: dc.float64[NI, NK],
    B: dc.float64[NK, NJ],
    C: dc.float64[NJ, NM],
    D: dc.float64[NM, NL],
):
    return A @ B @ C @ D
