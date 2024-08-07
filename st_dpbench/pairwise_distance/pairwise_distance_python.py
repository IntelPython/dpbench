# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.random as rnd

# constants used for input data generation
SEED = 7777777


# call numpy.random.uniform to generate input data
def gen_rand_data(nopt, dims, dtype=np.float64):
    rnd.seed(SEED)
    return (
        rnd.random((nopt, dims)).astype(dtype),
        rnd.random((nopt, dims)).astype(dtype),
    )


def gen_data(nopt, dims):
    X, Y = gen_rand_data(nopt, dims)
    return (X, Y, np.empty((nopt, nopt)))


# Naieve pairwise distance impl - take an array representing M points in N dimensions, and return the M x M matrix of Euclidean distances
def pairwise_distance_python(X1, X2, D):
    x1 = np.sum(np.square(X1), axis=1)
    x2 = np.sum(np.square(X2), axis=1)
    np.dot(X1, X2.T, D)
    D *= -2
    x3 = x1.reshape(x1.size, 1)
    np.add(D, x3, D)
    np.add(D, x2, D)
    np.sqrt(D, D)


X, Y, p_D = gen_data(1024, 3)

pairwise_distance_python(X, Y, p_D)

n_D = np.fromfile("D.bin").reshape(1024, 1024)

print("Python D = ", p_D)
print("SYCL D = ", n_D)

if np.allclose(n_D, p_D):
    print("Test succeeded\n")
else:
    print("Test failed\n")
