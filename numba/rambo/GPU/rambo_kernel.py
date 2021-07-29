#/usr/bin/env python
import base_rambo
import numpy,math
from numba import jit
import numba_dppy
import dpctl

@jit(nopython=True, fastmath=True)
def vectmultiply(a, b):
    c = a * b
    return c[..., 0] - c[..., 1] - c[..., 2] - c[..., 3]

def get_inputs(ecms, nevts):
    pa = numpy.array([ecms / 2., 0., 0., ecms / 2])
    pb = numpy.array([ecms / 2., 0., 0., -ecms / 2])

    input_particles = numpy.array([pa, pb])
    input_particles = numpy.repeat(input_particles[numpy.newaxis, ...], nevts, axis=0)

    return input_particles

@jit(nopython=True, fastmath=True)
def get_momentum_sum(inarray):
    return numpy.sum(inarray, axis=1)

@jit(nopython=True)
def get_combined_mass(inarray):
    sum = get_momentum_sum(inarray)
    return get_mass(sum)

@jit(nopython=True, fastmath=True)
def get_mass(inarray):
    mom2 = numpy.sum(inarray[..., 1:4]**2, axis=1)
    mass = numpy.sqrt(inarray[..., 0]**2 - mom2)
    return mass

@jit(nopython=True)
def gen_rand_data(nevts, nout):
    C1 = numpy.empty((nevts, nout))
    F1 = numpy.empty((nevts, nout))
    Q1 = numpy.empty((nevts, nout))

    numpy.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = numpy.random.rand()
            F1[i, j] = numpy.random.rand()
            Q1[i, j] = numpy.random.rand()*numpy.random.rand()

    return C1, F1, Q1


@numba_dppy.kernel
def get_output_mom2(C1, F1, Q1, output, nout):
    i = numba_dppy.get_global_id(0)
    for j in range(nout):
        C = 2.*C1[i, j]-1.
        S = math.sqrt(1 - C*C)
        F = 2.*math.pi*F1[i, j]
        Q = -math.log(Q1[i, j])

        output[i, j, 0] = Q
        output[i, j, 1] = Q*S*math.sin(F)
        output[i, j, 2] = Q*S*math.cos(F)
        output[i, j, 3] = Q*C


def call_ocl(nevts, nout):
    C1, F1, Q1 = gen_rand_data(nevts, nout)
    output = numpy.empty((nevts, nout, 4))
    
    with dpctl.device_context(base_rambo.get_device_selector()):
        get_output_mom2[nevts,numba_dppy.DEFAULT_LOCAL_SIZE](C1, F1, Q1, output, nout)

    return output

def GeneratePoints(ecms, nevts, nout):
    input_particles = get_inputs(ecms, nevts)

    input_mass = get_combined_mass(input_particles)

    output_particles = call_ocl(nevts, nout)

    output_mom_sum = get_momentum_sum(output_particles)
    output_mass = get_mass(output_mom_sum)

    G = output_mom_sum[..., 0] / output_mass
    G = numpy.repeat(G[..., numpy.newaxis], nout, axis=1)
    X = input_mass / output_mass
    X = numpy.repeat(X[..., numpy.newaxis], nout, axis=1)

    output_mass = numpy.repeat(output_mass[..., numpy.newaxis], 3, axis=1)

    B = numpy.zeros(output_mom_sum.shape)
    B[..., 1:4] = -output_mom_sum[..., 1:4] / output_mass
    B = numpy.repeat(B[:, numpy.newaxis, :], nout, axis=1)
    
    A = 1. / (1. + G)

    E = output_particles[..., 0]
    BQ = -1. * vectmultiply(B, output_particles)
    C1 = E + A * BQ
    C1 = numpy.repeat(C1[..., numpy.newaxis], 4, axis=2)
    C = output_particles + B * C1
    D = G * E + BQ
    output_particles[..., 0] = X * D
    output_particles[..., 1:4] = numpy.repeat(X[..., numpy.newaxis], 3, axis=2) * C[..., 1:4]

    return numpy.concatenate((input_particles, output_particles), axis=1)


def rambo(evt_per_calc):
    ng = 4
    outint = 1

    h = [[], [], [], []]
    nruns = int(outint / evt_per_calc) + 1
    for i in range(nruns):
        e = GeneratePoints(100, evt_per_calc, ng)
        for x in range(4):
            tmp = numpy.max(e[:, 2:5, x], axis=1)
            for entry in tmp:
                h[x].append(entry)


base_rambo.run("Rambo Numba", rambo)
