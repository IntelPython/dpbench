# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import numpy

def vectmultiply(a, b):
    c = a * b
    return c[..., 0] - c[..., 1] - c[..., 2] - c[..., 3]

def get_inputs(ecms,nevts):
    # input_particles = numpy.zeros([self.nevts,self.nin,4])
    pa = numpy.array([ecms / 2.,0.,0.,ecms / 2])
    pb = numpy.array([ecms / 2.,0.,0.,-ecms / 2])

    input_particles = numpy.array([pa,pb])
    input_particles = numpy.repeat(input_particles[numpy.newaxis, ...], nevts, axis=0)

    return input_particles

def get_momentum_sum(inarray):
    return numpy.sum(inarray, axis=1)


def get_combined_mass(inarray):
    sum = get_momentum_sum(inarray)
    return get_mass(sum)


def get_mass(inarray):
    mom2 = numpy.sum(inarray[..., 1:4]**2, axis=1)
    mass = numpy.sqrt(inarray[..., 0]**2 - mom2)
    return mass


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

def get_output_mom2(C1, F1, Q1, nevts, nout):
    output = numpy.empty((nevts, nout, 4))

    for i in range(nevts):
        for j in range(nout):
            C = 2.*C1[i, j]-1.
            S = numpy.sqrt(1 - numpy.square(C))
            F = 2.*numpy.pi*F1[i, j]
            Q = -numpy.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q*S*numpy.sin(F)
            output[i, j, 2] = Q*S*numpy.cos(F)
            output[i, j, 3] = Q*C

    return output

def generate_points(ecms, nevts, nout):
    C1, F1, Q1 = gen_rand_data(nevts, nout)
    output_particles = get_output_mom2(C1, F1, Q1, nevts, nout)

    return output_particles

    # input_particles = get_inputs(ecms, nevts)
    # input_mass = get_combined_mass(input_particles)

    # output_mom_sum = get_momentum_sum(output_particles)
    # output_mass = get_mass(output_mom_sum)

    # G = output_mom_sum[..., 0] / output_mass
    # G = numpy.repeat(G[..., numpy.newaxis], nout, axis=1)
    # X = input_mass / output_mass
    # X = numpy.repeat(X[..., numpy.newaxis], nout, axis=1)

    # output_mass = numpy.repeat(output_mass[..., numpy.newaxis], 3, axis=1)

    # B = numpy.zeros(output_mom_sum.shape)
    # B[..., 1:4] = -output_mom_sum[..., 1:4] / output_mass
    # B = numpy.repeat(B[:, numpy.newaxis, :], nout, axis=1)

    # A = 1. / (1. + G)

    # E = output_particles[..., 0]
    # BQ = -1. * vectmultiply(B, output_particles)
    # C1 = E + A * BQ
    # C1 = numpy.repeat(C1[..., numpy.newaxis], 4, axis=2)
    # C = output_particles + B * C1
    # D = G * E + BQ
    # output_particles[..., 0] = X * D
    # output_particles[..., 1:4] = numpy.repeat(X[..., numpy.newaxis], 3, axis=2) * C[..., 1:4]

    # return numpy.concatenate((input_particles, output_particles), axis=1)


def rambo_python(evt_per_calc):
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
