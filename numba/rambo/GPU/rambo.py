import numpy
import numba
import dpctl
import os

import base_rambo
from device_selector import get_device_selector

backend = os.getenv("NUMBA_BACKEND", "legacy")
if backend == "legacy":
    import numba as nb

    __njit = nb.jit(nopython=True, parallel=False)
    __fmjit = nb.jit(nopython=True, parallel=False, fastmath=True)
else:
    import numba_dpcomp as nb

    __njit = nb.njit(parallel=True, enable_gpu_pipeline=True)
    __fmjit = nb.njit(parallel=True, fastmath=True, enable_gpu_pipeline=True)


# @__njit
def gen_rand_data(nevts, nout):
    C1 = numpy.empty((nevts, nout))
    F1 = numpy.empty((nevts, nout))
    Q1 = numpy.empty((nevts, nout))

    numpy.random.seed(777)
    for i in numba.prange(nevts):
        for j in range(nout):
            C1[i, j] = numpy.random.rand()
            F1[i, j] = numpy.random.rand()
            Q1[i, j] = numpy.random.rand() * numpy.random.rand()

    return C1, F1, Q1


@__fmjit
def get_output_mom2(C1, F1, Q1, nevts, nout):
    output = numpy.empty((nevts, nout, 4))

    for i in numba.prange(nevts):
        for j in range(nout):
            C = 2.0 * C1[i, j] - 1.0
            S = numpy.sqrt(1 - numpy.square(C))
            F = 2.0 * numpy.pi * F1[i, j]
            Q = -numpy.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * numpy.sin(F)
            output[i, j, 2] = Q * S * numpy.cos(F)
            output[i, j, 3] = Q * C

    return output


def generate_points(ecms, nevts, nout):
    C1, F1, Q1 = gen_rand_data(nevts, nout)

    with dpctl.device_context(get_device_selector(is_gpu=True)):
        output_particles = get_output_mom2(C1, F1, Q1, nevts, nout)

    return output_particles


def rambo(evt_per_calc):
    ng = 4
    outint = 1

    h = [[], [], [], []]
    nruns = int(outint / evt_per_calc) + 1
    for i in range(nruns):
        e = generate_points(100, evt_per_calc, ng)

    return e
    # for x in range(4):
    #     tmp = numpy.max(e[:, 2:5, x], axis=1)
    #     for entry in tmp:
    #         h[x].append(entry)

    # fig,ax = plt.subplots(2,2,figsize=(10,12),dpi=80)
    # for i in range(len(ax)):
    #     for j in range(len(ax[i])):

    #         a = ax[i][j]

    #         data = h[i * len(ax[i]) + j]
    #         bins = range(0,100,1)
    #         a.hist(data,bins=bins)

    # plt.show()


base_rambo.run("Rambo Numba", rambo)
