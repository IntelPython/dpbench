# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Principle Component Analysis

Input
---------
data : array
       random regression problem

Output
-------
data: array
      transformation on the data using eigenvectors

evalues: array
         Eigen values

evectors: array
          Eigen vectors

Method
------
PCA implementation using covariance approach.
Step 1) Calculate covariance matrix
Step 2) Compute eigen values and eigen vectors from the covariance matrix
"""
