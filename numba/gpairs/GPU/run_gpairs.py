import base_gpairs
import numpy as np
import gaussian_weighted_pair_counts as gwpc
import numba_dppy
import dpctl
# from gpairs.pair_counter.tests.generate_test_data import (
#     DEFAULT_RBINS_SQUARED)

# DEFAULT_NBINS = 20
# DEFAULT_RMIN, DEFAULT_RMAX = 0.1, 50
# DEFAULT_RBINS = np.logspace(
#     np.log10(DEFAULT_RMIN), np.log10(DEFAULT_RMAX), DEFAULT_NBINS).astype(
#         np.float32)
# DEFAULT_RBINS_SQUARED = (DEFAULT_RBINS**2).astype(np.float32)

def run_gpairs(d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, result):
    blocks = 512

    with dpctl.device_context(base_gpairs.get_device_selector()):
        # gwpc.count_weighted_pairs_3d_intel[blocks, numba_dppy.DEFAULT_LOCAL_SIZE](
        #     d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
        #     d_rbins_squared, result)

        gwpc.count_weighted_pairs_3d_intel_ver2[d_x1.shape[0], numba_dppy.DEFAULT_LOCAL_SIZE](
            d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
            d_rbins_squared, result)
        
base_gpairs.run("Gpairs Dppy kernel",run_gpairs)
