# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import erf, exp, log, sqrt
from timeit import default_timer

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

now = default_timer


@dpex.kernel
def _black_scholes_kernel_test(
    item: kapi.Item, price, strike, t, rate, volatility, call, put
):
    dtype = price.dtype
    mr = -rate
    sig_sig_two = volatility * volatility * dtype.type(2)

    i = item.get_id(0)

    P = price[i]
    S = strike[i]
    T = t[i]

    a = log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = dtype.type(0.25) * z
    y = dtype.type(1.0) / sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = dtype.type(0.5) + dtype.type(0.5) * erf(w1)
    d2 = dtype.type(0.5) + dtype.type(0.5) * erf(w2)

    Se = exp(b) * S

    r = P * d1 - Se * d2
    call[i] = r
    put[i] = r - P + Se


def black_scholes_test(nopt, price, strike, t, rate, volatility, call, put):
    dpex.call_kernel(
        _black_scholes_kernel_test,
        kapi.Range(nopt),
        price,
        strike,
        t,
        rate,
        volatility,
        call,
        put,
    )


def initialize(nopt, seed):
    import numpy as np
    import numpy.random as default_rng

    dtype = np.dtype("f8")
    S0L = dtype.type(10.0)
    S0H = dtype.type(50.0)
    XL = dtype.type(10.0)
    XH = dtype.type(50.0)
    TL = dtype.type(1.0)
    TH = dtype.type(2.0)
    RISK_FREE = dtype.type(0.1)
    VOLATILITY = dtype.type(0.2)

    default_rng.seed(seed)
    price = default_rng.uniform(S0L, S0H, nopt).astype(dtype)
    strike = default_rng.uniform(XL, XH, nopt).astype(dtype)
    t = default_rng.uniform(TL, TH, nopt).astype(dtype)
    rate = RISK_FREE
    volatility = VOLATILITY
    call = np.zeros(nopt, dtype=dtype)
    put = -np.ones(nopt, dtype=dtype)

    return (price, strike, t, rate, volatility, call, put)


def copy_to_func():
    """Returns the copy-method that should be used
    for copying the benchmark arguments."""

    def _copy_to_func_impl(ref_array):
        import dpnp

        if ref_array.flags["C_CONTIGUOUS"]:
            order = "C"
        elif ref_array.flags["F_CONTIGUOUS"]:
            order = "F"
        else:
            order = "K"
        return dpnp.asarray(
            ref_array,
            dtype=ref_array.dtype,
            order=order,
            usm_type=None,
            sycl_queue=None,
        )

    return _copy_to_func_impl


nopt = 268435456
(price, strike, t, rate, volatility, call, put) = initialize(nopt, 777777)

price_d = copy_to_func()(price)
strike_d = copy_to_func()(strike)
t_d = copy_to_func()(t)
call_d = copy_to_func()(call)
put_d = copy_to_func()(put)

black_scholes_test(
    nopt, price_d, strike_d, t_d, rate, volatility, call_d, put_d
)

t0 = now()
black_scholes_test(
    nopt, price_d, strike_d, t_d, rate, volatility, call_d, put_d
)
t1 = now()

print("TIME: {:10.6f}".format((t1 - t0)), flush=True)
