// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include <CL/sycl.hpp>

using namespace sycl;

template <typename FpTy>
void gaussian_kernel_1(FpTy *m_device,
                       const FpTy *a_device,
                       int size,
                       int t,
                       sycl::nd_item<3> item_ct1)
{
    if (item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2) >=
        size - 1 - t)
        return;
    m_device[size * (item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2) + t + 1) +
             t] = a_device[size * (item_ct1.get_local_range().get(2) *
                                       item_ct1.get_group(2) +
                                   item_ct1.get_local_id(2) + t + 1) +
                           t] /
                  a_device[size * t + t];
}

template <typename FpTy>
void gaussian_kernel_2(FpTy *m_device,
                       FpTy *a_device,
                       FpTy *b_device,
                       int size,
                       int j1,
                       int t,
                       sycl::nd_item<3> item_ct1)
{
    if (item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2) >=
        size - 1 - t)
        return;
    if (item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range().get(1) >=
        size - t)
        return;

    int xidx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
    int yidx = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
               item_ct1.get_local_id(1);

    a_device[size * (xidx + 1 + t) + (yidx + t)] -=
        m_device[size * (xidx + 1 + t) + t] * a_device[size * t + (yidx + t)];
    if (yidx == 0) {
        b_device[xidx + 1 + t] -=
            m_device[size * (xidx + 1 + t) + (yidx + t)] * b_device[t];
    }
}
