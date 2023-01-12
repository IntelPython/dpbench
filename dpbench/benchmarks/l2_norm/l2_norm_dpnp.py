# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

def l2_norm(a, d):
    sq = dpnp.square(a)
    sum = sq.sum(axis=1)
    d[:] = dpnp.sqrt(sum)
