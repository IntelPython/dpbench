# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dpbench.benchmarks.gpairs.gpairs_sycl_native_ext.sycl_gpairs import sycl_gpairs 


 
def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
 
    results,totTime1 = sycl_gpairs(x1, y1, z1, w1, x2, y2, z2, w2, rbins)
    print("Kernel exe time= ", totTime1)
    return results
