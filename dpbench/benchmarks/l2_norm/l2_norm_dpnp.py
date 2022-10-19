# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

def l2_norm(a, d):
    dpnp.copyto(d, dpnp.linalg.norm(a, axis=1))
