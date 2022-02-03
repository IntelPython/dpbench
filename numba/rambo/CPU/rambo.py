#!/usr/bin/env python
# import matplotlib.pyplot as plt
import mkl_random
import numpy
import numba
import base_rambo


@numba.jit(nopython=True, fastmath=True)
def vectmultiply(a, b):
    c = a * b
    return c[..., 0] - c[..., 1] - c[..., 2] - c[..., 3]


# def __init__(nevts,nin,nout,ecms):
#     self.nevts = nevts
#     self.nin = nin
#     if self.nin != 2:
#         raise Exception('nin can only be 2 currently')
#     self.nout = nout
#     self.ecms = ecms
#     pi2log = numpy.log(numpy.pi/2.)
#     Z = [ 0, 0, pi2log ]
#     for k in range(nin,nout+1):
#         Z.append(Z[k-1]+pi2log-2.*numpy.log(k-2))
#     for k in range(nin,nout+1):
#         Z[k] = Z[k]-numpy.log(k-1)
#     self.Z_N = Z[nout]


def get_inputs(ecms, nevts):
    # input_particles = numpy.zeros([self.nevts,self.nin,4])
    pa = numpy.array([ecms / 2.0, 0.0, 0.0, ecms / 2])
    pb = numpy.array([ecms / 2.0, 0.0, 0.0, -ecms / 2])

    input_particles = numpy.array([pa, pb])
    input_particles = numpy.repeat(input_particles[numpy.newaxis, ...], nevts, axis=0)

    return input_particles


@numba.jit(nopython=True, fastmath=True)
def get_momentum_sum(inarray):
    return numpy.sum(inarray, axis=1)


@numba.jit(nopython=True)
def get_combined_mass(inarray):
    sum = get_momentum_sum(inarray)
    return get_mass(sum)


@numba.jit(nopython=True, fastmath=True)
def get_mass(inarray):
    mom2 = numpy.sum(inarray[..., 1:4] ** 2, axis=1)
    mass = numpy.sqrt(inarray[..., 0] ** 2 - mom2)
    return mass


def gen_rand_data(nevts, nout):
    rand = mkl_random.RandomState(seed=777, brng="MT2203")

    C1 = rand.rand(nevts, nout)
    F1 = rand.rand(nevts, nout)
    Q1 = rand.rand(nevts, nout) * rand.rand(nevts, nout)

    return C1, F1, Q1


@numba.jit(nopython=True, parallel=True, fastmath=True)
def get_output_mom2(C1, F1, Q1, nevts, nout):
    output = numpy.empty((nevts, nout, 4))

    for i in numba.prange(nevts):
        for j in range(nout):
            C = 2.0 * C1[i, j] - 1.0
            S = numpy.sqrt(1 - C**2)
            F = 2.0 * numpy.pi * F1[i, j]
            Q = -numpy.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * numpy.sin(F)
            output[i, j, 2] = Q * S * numpy.cos(F)
            output[i, j, 3] = Q * C

    return output


# def get_output_mom(self):
#     C = 2.*numpy.random.rand(self.nevts,self.nout)-1.
#     S = numpy.sqrt(1 - C**2)
#     F = 2.*numpy.pi*numpy.random.rand(self.nevts,self.nout)
#     Q = -numpy.log(numpy.random.rand(self.nevts,self.nout)*numpy.random.rand(self.nevts,self.nout))
#     output = numpy.zeros([self.nevts,self.nout,4])
#     output[...,0] = Q
#     output[...,1] = Q*S*numpy.sin(F)
#     output[...,2] = Q*S*numpy.cos(F)
#     output[...,3] = Q*C
#     return output


def generate_points(ecms, nevts, nout):
    input_particles = get_inputs(ecms, nevts)

    input_mass = get_combined_mass(input_particles)

    C1, F1, Q1 = gen_rand_data(nevts, nout)
    output_particles = get_output_mom2(C1, F1, Q1, nevts, nout)

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

    A = 1.0 / (1.0 + G)

    E = output_particles[..., 0]
    BQ = -1.0 * vectmultiply(B, output_particles)
    C1 = E + A * BQ
    C1 = numpy.repeat(C1[..., numpy.newaxis], 4, axis=2)
    C = output_particles + B * C1
    D = G * E + BQ
    output_particles[..., 0] = X * D
    output_particles[..., 1:4] = (
        numpy.repeat(X[..., numpy.newaxis], 3, axis=2) * C[..., 1:4]
    )

    return numpy.concatenate((input_particles, output_particles), axis=1)


def rambo(evt_per_calc):
    ng = 4
    outint = 1

    h = [[], [], [], []]
    nruns = int(outint / evt_per_calc) + 1
    for i in range(nruns):
        e = generate_points(100, evt_per_calc, ng)
        for x in range(4):
            tmp = numpy.max(e[:, 2:5, x], axis=1)
            for entry in tmp:
                h[x].append(entry)

    # fig,ax = plt.subplots(2,2,figsize=(10,12),dpi=80)
    # for i in range(len(ax)):
    #     for j in range(len(ax[i])):

    #         a = ax[i][j]

    #         data = h[i * len(ax[i]) + j]
    #         bins = range(0,100,1)
    #         a.hist(data,bins=bins)

    # plt.show()


base_rambo.run("Rambo Numba", rambo)
