# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import base_l2_distance_cupy
import cupy as cp


@cp.fuse(kernel_name="l2_distance")
def l2_distance(a, b):
    sub = a - b
    sq = cp.square(sub)
    sum = cp.sum(sq)
    d = cp.sqrt(sum)
    return d


base_l2_distance_cupy.run("Cupy", l2_distance)
