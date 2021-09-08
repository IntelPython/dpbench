import base_gpairs_cuda
import numpy as np
from numba import cuda
from gaussian_weighted_pair_counts import count_weighted_pairs_3d_cuda
import math

def run_gpairs(d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result):
    threads = 512
    blocks = math.ceil(d_x1.shape[0] / threads)

    count_weighted_pairs_3d_cuda[blocks, threads](
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
        d_rbins_squared, d_result)

base_gpairs_cuda.run("Gpairs Cuda kernel",run_gpairs)
