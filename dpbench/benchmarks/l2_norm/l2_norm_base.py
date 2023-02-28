# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def l2_norm(a, d):
    d[:] = np.linalg.norm(a, axis=1)
