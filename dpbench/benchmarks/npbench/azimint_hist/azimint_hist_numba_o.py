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

import numba as nb
import numpy as np


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def azimint_hist(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    histw = np.histogram(radius, npt, weights=data)[0]
    return histw / histu
