# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_bs_erf
from dpnp import log, exp
from base_bs_erf import erf, invsqrt

def black_scholes ( nopt, price, strike, t, rate, vol, call, put ):
	mr = -rate
	sig_sig_two = vol * vol * 2

	P = price
	S = strike
	T = t

	a = log(P / S)
	b = T * mr

	z = T * sig_sig_two
	c = 0.25 * z
	y = invsqrt(z)

	w1 = (a - b + c) * y
	w2 = (a - b - c) * y

	d1 = 0.5 + 0.5 * erf(w1)
	d2 = 0.5 + 0.5 * erf(w2)

	Se = exp(b) * S

	call[:] = P * d1 - Se * d2
	put[:] = call - P + Se

def black_scholes_dpctl(nopt, price, strike, t, rate, vol, call, put):
    with dpctl.device_context("opencl:gpu"):
        black_scholes( nopt, price, strike, t, rate, vol, call, put )

base_bs_erf.run("Numba@jit-numpy", black_scholes_dpctl, nparr=True, pass_args=True)
