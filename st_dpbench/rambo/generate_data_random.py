# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

# constants used for input data generation


# write input data to a file in binary format
def __dump_binary__(C1, F1, Q1):
    with open("C1.bin", "w") as fd:
        C1.tofile(fd)

    with open("F1.bin", "w") as fd:
        F1.tofile(fd)

    with open("Q1.bin", "w") as fd:
        Q1.tofile(fd)


def gen_rand_data(nevts, nout):
    C1 = np.empty((nevts, nout))
    F1 = np.empty((nevts, nout))
    Q1 = np.empty((nevts, nout))

    np.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = np.random.rand()
            F1[i, j] = np.random.rand()
            Q1[i, j] = np.random.rand() * np.random.rand()

    return (
        C1,
        F1,
        Q1,
        np.empty((nevts, nout, 4)),
    )


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(nevts, nout, dtype=np.float64):
    C1, F1, Q1, output = gen_rand_data(nevts, nout)
    __dump_binary__(C1, F1, Q1)


if __name__ == "__main__":
    gen_data_to_file(16777216, 4)
