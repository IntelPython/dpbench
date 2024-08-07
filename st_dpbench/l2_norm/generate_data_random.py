# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.random as rnd

# constants used for input data generation
SEED = 777777
DTYPE = np.float64


# write input data to a file in binary format
def __dump_binary__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd)


# write input data to a file in text format
def __dump_text__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd, "\n", "%s")


def gen_data(npoints, dims):
    import numpy as np
    import numpy.random as default_rng

    default_rng.seed(777777)

    return (
        default_rng.random((npoints, dims)).astype(np.float64),
        np.zeros(npoints).astype(np.float64),
    )


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(npoints, dims):
    a_data, d_data = gen_data(npoints, dims)
    __dump_binary__(a_data, "a_data.bin")
    # __dump_binary__(d_data, "d_data.bin")


if __name__ == "__main__":
    gen_data_to_file(536870912, 3)
