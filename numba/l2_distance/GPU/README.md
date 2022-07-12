# L2-distance

This document has instructions for how to run L2_distance workload Numba version.

## Arguments

| Argument | Default value     | Description     |
| ---------| ------------------| --------------- |
|--steps|10|Number of workload runs|
|--step|2|Data growth factor on each iteration|
|--size|2 ** 20|Initial data size|
|--repeat|1|Iterations inside measured region|
|--json|False|Output json data filename|
|-d|1|Data dimension|
|--test|False|Validation mode|
|--usm|False|Use USM Shared|

## Run atomics version

Use atomics from Oneapi source.

```bash
NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1 \
NUMBA_DPEX_LLVM_SPIRV_ROOT=/path/to/oneapi/compiler/version/linux/bin \
SYCL_DEVICE_FILTER=level_zero \
python l2_distance_kernel.py
```

## USM version

```bash
python l2_distance_kernel.py --usm
```

## Testing

```bash
python l2_distance_kernel.py --test
```
