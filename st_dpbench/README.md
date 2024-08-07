<!--
SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation

SPDX-License-Identifier: Apache-2.0
-->

# ST-DPBench - Standalone benchmarks to evaluate Data-Parallel Extensions for Python

This directory contains standalone DPC++ and numba-dpex implementations of DPBench workloads. These implementations are meant be executed independent of dpbench infrastructure.

## Setting up the benchmarks

1. Install DPC++ compiler.

2. Setup conda enviroment
   > conda create -n st_dpbench
   > conda activate st_dpbench
   > conda install -c dppy/label/dev -c intel numba-dpex scikit-learn

3. Refer to README in each workload directory for specific instructions on how to run the workload.
