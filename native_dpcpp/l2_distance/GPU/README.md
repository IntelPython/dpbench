# L2-distance

This document has instructions for how to run L2_distance workload DPC++ version.

## Run workload

Kernel is based on the Reduction method by default.

```bash
python base_l2_distane.py
```

## Arguments

| Argument | Default value     | Description     | 
| ---------| ------------------| --------------- | 
|--steps|10|Number of workload runs|
|--step|2|Data growth factor on each iteration|
|--size|2 ** 20|Initial data size|
|--repeat|1|Iterations inside measured region|
|-d|1|Data dimension. Currently default value is the only option|
|--test|False|Validation mode|
|--usm|False|Use USM Shared|
|--atomic|False|Kernel based on atomics|

## Testing

```bash
python base_l2_distance.py --test
```
