# L2-distance

This document has instructions for how to run L2_distance workload Numba version.

## Atomics version
Use atomics from Oneapi source.
```bash
NUMBA_DPPY_ACTIVATE_ATOMCIS_FP_NATIVE=1 NUMBA_DPPY_LLVM_SPIRV_ROOT=/path/to/oneapi/compiler/version/linux/bin SYCL_DEVICE_FILTER=level_zero python l2_distance_kernel.py
```
