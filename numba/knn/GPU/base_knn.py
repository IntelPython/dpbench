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

import argparse
import sys, os, json
import numpy as np
import numpy.random as rnd

#from knn_python import knn_python
from dpbench_datagen.knn import gen_data_x, gen_data_y
import dpctl
import dpctl.tensor as dpt

from dpbench_python.knn.knn_python import knn_python

DATA_DIM = 16
SEED = 7777777
CLASSES_NUM = 3
TRAIN_DATA_SIZE = 2 ** 10
n_neighbors = 5

try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0), t1 - t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range


###############################################
def get_device_selector(is_gpu=True):
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if os.environ.get('SYCL_DEVICE_FILTER') is None or os.environ.get('SYCL_DEVICE_FILTER') == "opencl":
        return "opencl:" + device_selector

    if os.environ.get('SYCL_DEVICE_FILTER') == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get('SYCL_DEVICE_FILTER')


def gen_data_usm(nopt):
    # init numpy obj
    x_train, y_train = gen_data_x(TRAIN_DATA_SIZE, seed=0, dim=DATA_DIM), \
                       gen_data_y(TRAIN_DATA_SIZE, CLASSES_NUM, seed=0)
    x_test = gen_data_x(nopt, seed=777777, dim=DATA_DIM)

    predictions = np.empty(nopt)
    queue_neighbors_lst = np.empty((nopt, n_neighbors, 2))
    votes_to_classes_lst = np.zeros((nopt, CLASSES_NUM))

    with dpctl.device_context(get_device_selector()) as gpu_queue:
        train_usm = dpt.usm_ndarray(x_train.shape, dtype=x_train.dtype, buffer="device",
                                    buffer_ctor_kwargs={"queue": gpu_queue})
        train_labels_usm = dpt.usm_ndarray(y_train.shape, dtype=y_train.dtype, buffer="device",
                                           buffer_ctor_kwargs={"queue": gpu_queue})
        test_usm = dpt.usm_ndarray(x_test.shape, dtype=x_test.dtype, buffer="device",
                                   buffer_ctor_kwargs={"queue": gpu_queue})
        predictions_usm = dpt.usm_ndarray(predictions.shape, dtype=predictions.dtype, buffer="device",
                                          buffer_ctor_kwargs={"queue": gpu_queue})
        queue_neighbors_lst_usm = dpt.usm_ndarray(queue_neighbors_lst.shape, dtype=queue_neighbors_lst.dtype,
                                                  buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        votes_to_classes_lst_usm = dpt.usm_ndarray(votes_to_classes_lst.shape, dtype=votes_to_classes_lst.dtype,
                                                   buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})

    train_usm.usm_data.copy_from_host(x_train.reshape((-1)).view("|u1"))
    train_labels_usm.usm_data.copy_from_host(y_train.reshape((-1)).view("|u1"))
    test_usm.usm_data.copy_from_host(x_test.reshape((-1)).view("|u1"))
    predictions_usm.usm_data.copy_from_host(predictions.reshape((-1)).view("|u1"))
    queue_neighbors_lst_usm.usm_data.copy_from_host(queue_neighbors_lst.reshape((-1)).view("|u1"))
    votes_to_classes_lst_usm.usm_data.copy_from_host(votes_to_classes_lst.reshape((-1)).view("|u1"))

    return train_usm, train_labels_usm, test_usm, predictions_usm, queue_neighbors_lst_usm, votes_to_classes_lst_usm


##############################################

def run(name, alg, sizes=10, step=2, nopt=2 ** 10):
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=sizes,
                        help='Number of steps')
    parser.add_argument('--step', type=int, default=step,
                        help='Factor for each step')
    parser.add_argument('--size', type=int, default=nopt,
                        help='Initial data size')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Iterations inside measured region')
    parser.add_argument('--text', default='', help='Print with each result')
    parser.add_argument('--test', required=False, action='store_true',
                        help="Check for correctness by comparing output with naieve Python version")
    parser.add_argument('--json', required=False, default=__file__.replace('py', 'json'),
                        help="output json data filename")
    parser.add_argument('--usm', required=False, action='store_true', help="Use USM Shared or pure numpy")

    args = parser.parse_args()
    nopt = args.size
    repeat = args.repeat
    train_data_size = TRAIN_DATA_SIZE

    output = {}
    output['name'] = name
    output['sizes'] = sizes
    output['step'] = step
    output['repeat'] = repeat
    output['randseed'] = SEED
    output['metrics'] = []

    rnd.seed(SEED)

    if args.test:
        x_train, y_train = gen_data_x(train_data_size, seed=0, dim=DATA_DIM), \
                           gen_data_y(train_data_size, CLASSES_NUM, seed=0)
        x_test = gen_data_x(nopt, seed=777777, dim=DATA_DIM)
        p_predictions = np.empty(nopt)
        p_queue_neighbors_lst = np.empty((nopt, n_neighbors, 2))
        p_votes_to_classes_lst = np.zeros((nopt, CLASSES_NUM))

        knn_python(x_train, y_train, x_test, n_neighbors, CLASSES_NUM, TRAIN_DATA_SIZE, nopt, p_predictions,
                  p_queue_neighbors_lst, p_votes_to_classes_lst, DATA_DIM)

        if args.usm is True:  # test usm feature
            train_usm, train_labels_usm, test_usm, predictions_usm, queue_neighbors_lst_usm, votes_to_classes_lst_usm = \
            gen_data_usm(nopt)
            alg(train_usm, train_labels_usm, test_usm, n_neighbors, CLASSES_NUM, nopt, TRAIN_DATA_SIZE, predictions_usm,
                    queue_neighbors_lst_usm, votes_to_classes_lst_usm)

            n_predictions = np.empty(nopt)
            predictions_usm.usm_data.copy_to_host(n_predictions.view("u1"))

        else:
            n_predictions = np.empty(nopt)
            n_queue_neighbors_lst = np.empty((nopt, n_neighbors, 2))
            n_votes_to_classes_lst = np.zeros((nopt, CLASSES_NUM))

            alg(x_train, y_train, x_test, n_neighbors, CLASSES_NUM, TRAIN_DATA_SIZE, nopt, n_predictions,
             n_queue_neighbors_lst, n_votes_to_classes_lst, DATA_DIM)

        if np.allclose(n_predictions, p_predictions):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    with open('perf_output.csv', 'w', 1) as fd, open("runtimes.csv", 'w', 1) as fd2:

        for _ in xrange(args.steps):
            sys.stdout.flush()

            if args.usm is True:
                x_train, y_train, x_test, predictions, queue_neighbors_lst, votes_to_classes_lst = \
                    gen_data_usm(nopt)
            else:
                x_train, y_train = gen_data_x(TRAIN_DATA_SIZE, seed=0, dim=DATA_DIM), \
                                   gen_data_y(TRAIN_DATA_SIZE, CLASSES_NUM, seed=0)

                x_test = gen_data_x(nopt, seed=777777, dim=DATA_DIM)
                predictions = np.empty(nopt)
                queue_neighbors_lst = np.empty((nopt, n_neighbors, 2))
                votes_to_classes_lst = np.zeros((nopt, CLASSES_NUM))

            alg(x_train, y_train, x_test, n_neighbors, CLASSES_NUM, nopt, TRAIN_DATA_SIZE, predictions,
                queue_neighbors_lst, votes_to_classes_lst, DATA_DIM)  # warmup

            t0 = now()
            for _ in xrange(repeat):
                alg(x_train, y_train, x_test, n_neighbors, CLASSES_NUM, nopt, TRAIN_DATA_SIZE, predictions,
                    queue_neighbors_lst, votes_to_classes_lst, DATA_DIM)
            mops, time = get_mops(t0, now(), nopt)

            result_mops = mops * repeat
            fd.write('{},{}\n'.format(nopt, result_mops))
            fd2.write('{},{}\n'.format(nopt, time))

            print(
                "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, mops * repeat, time),
                flush=True)
            output['metrics'].append((nopt, mops, time))

            nopt *= args.step
            repeat = max(repeat - args.step, 1)
    json.dump(output, open(args.json, 'w'), indent=2, sort_keys=True)
