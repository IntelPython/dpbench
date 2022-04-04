# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import contextlib
import dpctl
import os
import numpy as np

import base_l2_distance
from device_selector import get_device_selector

def get_jit(is_gpu, backend):
    if backend == "legacy":
        import numba as nb

        __njit = nb.njit(parallel=True, fastmath=True)
    elif backend == "mlir":
        import numba_dpcomp as nb

        __njit = nb.njit(parallel=True, fastmath=True, enable_gpu_pipeline=is_gpu)
    else:
        __njit = lambda fn : fn

def make_jit(fn, is_gpu, banckend):
    return get_jit(is_gpu, banckend)(fn)

def run_kernel(fn, is_gpu, bakend, *args):
    with contextlib.ExitStack() as stack:
        if is_gpu:
            stack.enter_context(dpctl.device_context(get_device_selector(is_gpu=True)))
        return make_jit(fn, is_gpu, bakend))(*args)

#############################


def l2_distance_kernel(a, b):
    sub = a - b  # this line is offloaded
    sq = np.square(sub)  # this line is offloaded
    sum = np.sum(sq)
    d = np.sqrt(sum)
    return d

def l2_distance(is_gpu, bakend, a, b, _):
    return run_kernel(l2_distance_kernel, is_gpu, bakend, a, b, _)
    
    #with contextlib.ExitStack() as stack:
    #    if is_gpu:
    #        stack.enter_context(dpctl.device_context(get_device_selector(is_gpu=True)))
    #    return make_jit(l2_distance_kernel, is_gpu, bakend))(a, b)

base_l2_distance.run("l2 distance", l2_distance)
