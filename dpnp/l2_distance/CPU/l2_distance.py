# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_l2_distance
import dpnp as np

def l2_distance(a,b):
    sub = a-b
    sq = np.square(sub)
    sum = np.sum(sq)
    d = np.sqrt(sum)
    return d

def l2_distance_dpctl(a,b):
    with dpctl.device_context("opencl:cpu"):
        l2_distance(a,b)

base_l2_distance.run("l2 distance", l2_distance_dpctl)
