# SPDX-FileCopyrightText: 2014 Jérôme Kieffer et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
high performance azimuthal integration on gpu, 2014. In Proceedings of the
7th European Conference on Python in Science (EuroSciPy 2014).
"""

import dace as dc
import numpy as np

N, npt = (dc.symbol(s, dtype=dc.int64) for s in ("N", "npt"))


@dc.program
def azimint_naive(data: dc.float64[N], radius: dc.float64[N]):
    # rmax = radius.max()
    rmax = np.amax(radius)
    res = np.zeros((npt,), dtype=np.float64)  # Fix in np.full
    for i in range(npt):
        # for i in dc.map[0:npt]:  # Optimization
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        # values_r12 = data[mask_r12]
        # res[i] = np.mean(values_r12)
        on_values = 0
        tmp = np.float64(0)
        for j in dc.map[0:N]:
            if mask_r12[j]:
                tmp += data[j]
                on_values += 1
        res[i] = tmp / on_values
    return res
