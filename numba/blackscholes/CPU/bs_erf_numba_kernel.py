# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

from math import erf, exp, log, sqrt

import base_bs_erf
import dpctl
import numba_dpex
from device_selector import get_device_selector


# blackscholes implemented using dpex.kernel
@numba_dpex.kernel
def black_scholes(nopt, price, strike, t, rate, vol, call, put):
    mr = -rate
    sig_sig_two = vol * vol * 2

    i = numba_dpex.get_global_id(0)

    P = price[i]
    S = strike[i]
    T = t[i]

    a = log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1.0 / sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * erf(w1)
    d2 = 0.5 + 0.5 * erf(w2)

    Se = exp(b) * S

    r = P * d1 - Se * d2
    call[i] = r
    put[i] = r - P + Se


def black_scholes_driver(nopt, price, strike, t, rate, vol, call, put):
    # offload blackscholes computation to CPU (toggle level0 or opencl driver).
    with dpctl.device_context(get_device_selector(is_gpu=False)):
        black_scholes[nopt, numba_dpex.DEFAULT_LOCAL_SIZE](
            nopt, price, strike, t, rate, vol, call, put
        )


# call the run function to setup input data and performance data infrastructure
base_bs_erf.run("Numba@jit-loop-par", black_scholes_driver)
