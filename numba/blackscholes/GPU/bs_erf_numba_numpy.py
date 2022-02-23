# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_bs_erf
from device_selector import get_device_selector
import numpy as np
import os
from numpy import log, exp, sqrt
from math import erf

backend = os.getenv("NUMBA_BACKEND", "legacy")
if backend == "legacy":
    import numba as nb

    __njit = nb.njit(parallel=True, fastmath=True)
    __vectorize = nb.vectorize(nopython=True)
else:
    import numba_dpcomp as nb

    __njit = nb.njit(parallel=True, fastmath=True, enable_gpu_pipeline=True)
    __vectorize = nb.vectorize(nopython=True, enable_gpu_pipeline=True)

# Numba does know erf function from numpy or scipy
@__vectorize
def nberf(x):
    return erf(x)

# blackscholes implemented using numpy function calls
@__njit
def black_scholes_kernel(price, strike, t, rate, vol, call, put):
    mr = -rate
    sig_sig_two = vol * vol * 2

    P = price
    S = strike
    T = t

    a = log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1.0 / sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * nberf(w1)
    d2 = 0.5 + 0.5 * nberf(w2)

    Se = exp(b) * S

    r = P * d1 - Se * d2
    call[:] = r  # temporary `r` is necessary for faster `put` computation
    put[:] = r - P + Se


def black_scholes(nopt, price, strike, t, rate, vol, call, put):
    # offload blackscholes computation to GPU (toggle level0 or opencl driver).
    with dpctl.device_context(get_device_selector(is_gpu=True)):
        black_scholes_kernel(price, strike, t, rate, vol, call, put)


# call the run function to setup input data and performance data infrastructure
base_bs_erf.run("Numba@jit-numpy", black_scholes)
