import base_gpairs_cuda
import numpy as np
from numba import cuda
from gaussian_weighted_pair_counts import count_weighted_pairs_3d_cuda

DEFAULT_NBINS = 20
DEFAULT_RMIN, DEFAULT_RMAX = 0.1, 50
DEFAULT_RBINS = np.logspace(
    np.log10(DEFAULT_RMIN), np.log10(DEFAULT_RMAX), DEFAULT_NBINS).astype(
        np.float32)
DEFAULT_RBINS_SQUARED = (DEFAULT_RBINS**2).astype(np.float32)

def run_gpairs(d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared):
    blocks = 512
    threads = 512

    result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1]
    result = result.astype(np.float32)

    d_result = cuda.device_array_like(result.astype(np.float64))
    count_weighted_pairs_3d_cuda[blocks, threads](
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
        d_rbins_squared, d_result)

base_gpairs_cuda.run("Gpairs Cuda kernel",run_gpairs)
