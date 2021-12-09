import base_gpairs
import numpy as np
import gaussian_weighted_pair_counts as gwpc
# import numba_dppy
from numba_dpcomp.mlir.kernel_impl import DEFAULT_LOCAL_SIZE
import dpctl


def run_gpairs(
    d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    blocks = 512

    with dpctl.device_context(base_gpairs.get_device_selector()):
        gwpc.count_weighted_pairs_3d_intel[
            d_x1.shape[0], DEFAULT_LOCAL_SIZE
        ](d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result)


base_gpairs.run("Gpairs Dppy kernel", run_gpairs)
