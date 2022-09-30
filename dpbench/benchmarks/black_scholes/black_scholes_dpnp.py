# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

invsqrt = lambda x: np.true_divide(1.0, np.sqrt(x))


def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    P = price
    S = strike
    T = t

    a = dpnp.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = invsqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * dpnp.erf(w1)
    d2 = 0.5 + 0.5 * dpnp.erf(w2)

    Se = dpnp.exp(b) * S

    call[:] = P * d1 - Se * d2
    put[:] = call - P + Se
