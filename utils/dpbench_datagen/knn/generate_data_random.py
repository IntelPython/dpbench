import numpy as np
try:
    import numpy.random_intel as rnd
    numpy_ver="Intel"
except:
    import numpy.random as rnd
    numpy_ver="regular"

DATA_DIM = 16
#constants used for input data generation
SEED_TEST = 777777
SEED_TRAIN = 0

CLASSES_NUM = 3
TRAIN_DATA_SIZE = 2**10


#write input data to a file in binary format
def __dump_binary__(data_array, file_name):
    with open(file_name, 'w') as fd:
        data_array.tofile(fd)

#write input data to a file in text format
def __dump_text__(data_array, file_name):
    with open(file_name, 'w') as fd:
        data_array.tofile(fd, '\n', '%s')


def gen_data_x(nopt, type=np.float64, seed=SEED_TRAIN, dim=DATA_DIM):
    rnd.seed(seed)
    data = rnd.rand(nopt, dim)
    return data.astype(type)


def gen_data_y(nopt, classes_num=CLASSES_NUM, seed=SEED_TRAIN):
    rnd.seed(seed)
    data = rnd.randint(classes_num, size=nopt)
    return data


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(nopt=2**10, dtype=np.float64, dim=DATA_DIM):
    x_train, y_train = gen_data_x(TRAIN_DATA_SIZE, dtype, dim=dim), gen_data_y(TRAIN_DATA_SIZE, CLASSES_NUM)
    x_test = gen_data_x(nopt, dtype, seed=SEED_TEST, dim=dim)

    __dump_binary__(x_train, "x_train.bin")
    __dump_binary__(y_train, "y_train.bin")
    __dump_binary__(x_test, "x_test.bin")
