import numpy
import math
import dpctl

import base_rambo
from device_selector import get_device_selector

from numba_dppy import kernel, get_global_id, atomic, DEFAULT_LOCAL_SIZE
import numba_dppy

def gen_rand_data(nevts, nout):
    C1 = numpy.empty((nevts, nout))
    F1 = numpy.empty((nevts, nout))
    Q1 = numpy.empty((nevts, nout))

    numpy.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = numpy.random.rand()
            F1[i, j] = numpy.random.rand()
            Q1[i, j] = numpy.random.rand() * numpy.random.rand()

    return C1, F1, Q1


@kernel
def get_output_mom2(C1, F1, Q1, output, nout):
    i = numba_dppy.get_global_id(0)
    for j in range(nout):
        C = 2.0 * C1[i, j] - 1.0
        S = math.sqrt(1 - C * C)
        F = 2.0 * math.pi * F1[i, j]
        Q = -math.log(Q1[i, j])

        output[i, j, 0] = Q
        output[i, j, 1] = Q * S * math.sin(F)
        output[i, j, 2] = Q * S * math.cos(F)
        output[i, j, 3] = Q * C


# memory allocation in the kernel??
def GeneratePoints(nevts, nout):
    # kernel is on CPU
    C1, F1, Q1 = gen_rand_data(nevts, nout)

    output = numpy.empty((nevts, nout, 4))

    with dpctl.device_context(get_device_selector(is_gpu=True)):
        get_output_mom2[nevts, DEFAULT_LOCAL_SIZE](C1, F1, Q1, output, nout)

    return output


# very strange kernel
def rambo(evt_per_calc):
    ng = 4
    outint = 1

    nruns = int(outint / evt_per_calc) + 1
    for i in range(nruns):
        e = GeneratePoints(evt_per_calc, ng)
    return e


base_rambo.run("Rambo Numba", rambo)
