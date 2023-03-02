# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def l2_norm(a, d):
    d[:] = np.linalg.norm(a, axis=1)
