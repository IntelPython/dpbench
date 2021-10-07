import numpy as np

# constants used for input data generation
DTYPE = np.float32


# write input data to a file in binary format
def __dump_binary__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd)


# write input data to a file in text format
def __dump_text__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd, "\n", "%s")


def gen_matrix(size, dtype=DTYPE):
    """
    Example of target matrix m with size = 4

    10.0 9.9 9.8 9.7
    9.9 10.0 9.9 9.8
    9.8 9.9 10.0 9.9
    9.7 9.8 9.9 10.0

    """

    m = np.empty(size * size, dtype=dtype)

    lamda = -0.01
    coef = np.empty(2 * size - 1)

    for i in range(size):
        coef_i = 10 * np.exp(lamda * i)
        j = size - 1 + i
        coef[j] = coef_i
        j = size - 1 - i
        coef[j] = coef_i

    for i in range(size):
        for j in range(size):
            m[i * size + j] = coef[size - 1 - i + j]

    return m


def gen_vec(size, value, dtype=DTYPE):
    return np.full(size, value, dtype=dtype)


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(size, value, dtype=DTYPE):
    m_data = gen_matrix(size, dtype=dtype)
    v_data = gen_vec(size, value, dtype=dtype)
    __dump_binary__(m_data, "m_data.bin")
    __dump_binary__(v_data, "v_data.bin")
