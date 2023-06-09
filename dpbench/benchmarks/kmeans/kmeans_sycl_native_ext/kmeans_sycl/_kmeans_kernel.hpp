// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <algorithm>
#include <limits>

#include <iostream>
#include <stdio.h>

#include <string>
#include <vector>

template <class T>
using DeviceMem = std::unique_ptr<T[], std::function<void(T *)>>;

template <class T>
DeviceMem<T> malloc_device_mem(int count, cl::sycl::queue &queue)
{
    return DeviceMem<T>(sycl::malloc_device<T>(count, queue),
                        [queue](T *mem) { sycl::free(mem, queue); });
}

template <class NumT, class DenT> NumT DivUp(NumT numerator, DenT denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <class VT, class BT> VT Align(VT value, BT base)
{
    return base * DivUp(value, base);
}

template <class T>
void print_buffer(sycl::queue queue, T *data, int size, std::string name)
{
    std::vector<T> host_data(size);

    queue.copy(data, host_data.data(), size).wait();

    std::cout << name << ": ";
    for (auto &&v : host_data)
        std::cout << v << " ";

    std::cout << std::endl;
}

template <typename FpTy, typename IntTy, int Dims> class LloydSingleStep;
template <typename FpTy, typename IntTy, int Dims> class CalcDifference;
template <typename FpTy, typename IntTy, int Dims> class UpdateLabels;

template <typename FpTy, typename IntTy, int Dims>
void kmeans_impl(sycl::queue q,
                 FpTy *arrayP,
                 IntTy *arrayPclusters,
                 FpTy *arrayC,
                 IntTy *arrayCnumpoint,
                 size_t niters,
                 size_t npoints,
                 size_t ncentroids,
                 FpTy tolerance)
{
    FpTy diff_cpu = std::numeric_limits<FpTy>::infinity();
    int local_size = std::min(
        1024, int(q.get_device()
                      .get_info<sycl::info::device::max_work_group_size>()));
    int local_copies =
        std::min(4, std::max(1, int(DivUp(local_size, ncentroids))));
    bool early_exit = false;
    int WorkPI = 8;
    int global_size = Align(DivUp(npoints, WorkPI), local_size);

    auto newArrayCPtr = malloc_device_mem<FpTy>(ncentroids * Dims, q);
    auto newArrayC = newArrayCPtr.get();
    q.fill<FpTy>(newArrayC, 0, ncentroids * Dims).wait();

    auto newCNumpointsPtr = malloc_device_mem<int>(ncentroids, q);
    auto newCNumpoints = newCNumpointsPtr.get();
    q.fill<int>(newCNumpoints, 0, ncentroids).wait();

    FpTy host_diff = std::numeric_limits<FpTy>::infinity();

    auto diffPtr = malloc_device_mem<FpTy>(1, q);
    auto diff = diffPtr.get();

    q.copy(arrayP, arrayC, ncentroids * Dims).wait();

    for (int i = 0; i < niters; i++) {
        bool last = i == (niters - 1);

        if (host_diff < tolerance) {
            early_exit = true;
            break;
        }

        q.submit([&](sycl::handler &cgh) {
             auto localCentroinds = sycl::local_accessor<FpTy, 2>(
                 sycl::range<2>(Dims, ncentroids), cgh);
             auto localNewCentroinds = sycl::local_accessor<FpTy, 3>(
                 sycl::range<3>(local_copies, Dims, ncentroids), cgh);
             auto localNewNPoints = sycl::local_accessor<int, 2>(
                 sycl::range<2>(local_copies, ncentroids), cgh);

             cgh.parallel_for<LloydSingleStep<FpTy, IntTy, Dims>>(
                 sycl::nd_range<1>(sycl::range(global_size),
                                   sycl::range(local_size)),
                 [=](sycl::nd_item<1> item) {
                     auto group = item.get_group();
                     auto grid = item.get_group_linear_id();
                     auto lid = item.get_local_linear_id();

                     for (int i = lid; i < ncentroids * Dims; i += local_size) {
                         localCentroinds[i % Dims][i / Dims] = arrayC[i];
                         for (int lc = 0; lc < local_copies; ++lc) {
                             localNewCentroinds[lc][i % Dims][i / Dims] = 0;
                         }
                     }

                     for (int i = lid; i < ncentroids; i += local_size) {
                         for (int lc = 0; lc < local_copies; ++lc) {
                             localNewNPoints[lc][i] = 0;
                         }
                     }

                     sycl::group_barrier(group, sycl::memory_scope::work_group);

                     for (int i = 0; i < WorkPI; ++i) {
                         int point_id =
                             grid * WorkPI * local_size + i * local_size + lid;
                         if (point_id < npoints) {
                             FpTy localP[Dims];
                             for (int d = 0; d < Dims; ++d)
                                 localP[d] = arrayP[point_id * Dims + d];

                             auto minor_distance =
                                 std::numeric_limits<FpTy>::infinity();
                             int nearest_centroid = 0;
                             for (size_t c = 0; c < ncentroids; c++) {
                                 FpTy sq_sum = 0;
                                 for (size_t d = 0; d < Dims; d++) {
                                     FpTy dist =
                                         localP[d] - localCentroinds[d][c];
                                     sq_sum += dist * dist;
                                 }

                                 if (minor_distance > sq_sum) {
                                     minor_distance = sq_sum;
                                     nearest_centroid = c;
                                 }
                             }

                             auto lc = lid % local_copies;

                             for (int d = 0; d < Dims; ++d) {
                                 sycl::atomic_ref<
                                     FpTy, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group>
                                     centroid_d(
                                         localNewCentroinds[lc][d]
                                                           [nearest_centroid]);
                                 centroid_d += localP[d];
                             }

                             sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                              sycl::memory_scope::work_group>
                                 centroid_num_points(
                                     localNewNPoints[lc][nearest_centroid]);
                             centroid_num_points += 1;

                             if (last)
                                 arrayPclusters[point_id] = nearest_centroid;
                         }
                     }
                     sycl::group_barrier(group, sycl::memory_scope::work_group);

                     for (int i = lid; i < ncentroids * Dims; i += local_size) {
                         int local_centroid_npoints = 0;
                         FpTy local_centroid_d = 0;
                         for (int lc = 0; lc < local_copies; ++lc)
                             local_centroid_d +=
                                 localNewCentroinds[lc][i % Dims][i / Dims];

                         sycl::atomic_ref<FpTy, sycl::memory_order::relaxed,
                                          sycl::memory_scope::device>
                             centroid_d(newArrayC[i]);
                         centroid_d += (local_centroid_d);
                     }

                     for (int i = lid; i < ncentroids; i += local_size) {
                         int local_centroid_npoints = 0;
                         for (int lc = 0; lc < local_copies; ++lc)
                             local_centroid_npoints += localNewNPoints[lc][i];

                         sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                          sycl::memory_scope::device>
                             centroid_npoints(newCNumpoints[i]);
                         centroid_npoints += (local_centroid_npoints);
                     }
                 });
         }).wait();

        q.submit([&](sycl::handler &cgh) {
             int size = std::min(local_size, int(ncentroids));
             cgh.parallel_for<CalcDifference<FpTy, IntTy, Dims>>(
                 sycl::nd_range<1>(sycl::range(size), sycl::range(size)),
                 [=](sycl::nd_item<1> item) {
                     auto lid = item.get_local_id();
                     auto group = item.get_group();

                     FpTy max_distance = 0;
                     for (int i = lid; i < ncentroids; i += size) {
                         int numpoints = newCNumpoints[i];
                         arrayCnumpoint[i] = numpoints;
                         newCNumpoints[i] = 0;
                         FpTy distance = 0;
                         for (int d = 0; d < Dims; ++d) {
                             auto d0 = arrayC[i * Dims + d];
                             auto d1 = newArrayC[i * Dims + d];
                             newArrayC[i * Dims + d] = 0;

                             d1 = numpoints > 0 ? d1 / numpoints : d0;

                             distance += d0 * d0 - d1 * d1;

                             arrayC[i * Dims + d] = d1;
                         }

                         max_distance = std::max(max_distance, distance);
                     }

                     sycl::group_barrier(group, sycl::memory_scope::work_group);

                     max_distance = sycl::reduce_over_group(group, max_distance,
                                                            sycl::maximum());

                     if (lid == 0) {
                         diff[0] = std::sqrt(max_distance);
                     }
                 });
         }).wait();

        q.copy(diff, &host_diff, 1).wait();
    }

    if (early_exit) {
        q.submit([&](sycl::handler &cgh) {
             auto localCentroinds = sycl::local_accessor<FpTy, 2>(
                 sycl::range<2>(Dims, ncentroids), cgh);

             cgh.parallel_for<UpdateLabels<FpTy, IntTy, Dims>>(
                 sycl::nd_range<1>(sycl::range(global_size),
                                   sycl::range(local_size)),
                 [=](sycl::nd_item<1> item) {
                     auto group = item.get_group();
                     auto grid = item.get_group_linear_id();
                     auto lid = item.get_local_linear_id();

                     for (int i = lid; i < ncentroids * Dims; i += local_size) {
                         localCentroinds[i % Dims][i / Dims] = arrayC[i];
                     }

                     sycl::group_barrier(group, sycl::memory_scope::work_group);

                     for (int i = 0; i < WorkPI; ++i) {
                         int point_id =
                             grid * WorkPI * local_size + i * local_size + lid;
                         if (point_id < npoints) {
                             FpTy localP[Dims];
                             for (int d = 0; d < Dims; ++d)
                                 localP[d] =
                                     arrayP[grid * WorkPI * local_size * Dims +
                                            i * local_size * Dims + lid * Dims +
                                            d];

                             auto minor_distance =
                                 std::numeric_limits<FpTy>::infinity();
                             int nearest_centroid = 0;
                             for (size_t c = 0; c < ncentroids; c++) {
                                 FpTy sq_sum = 0;
                                 for (size_t d = 0; d < Dims; d++) {
                                     FpTy dist =
                                         localP[d] - localCentroinds[d][c];
                                     sq_sum += dist * dist;
                                 }

                                 if (minor_distance > sq_sum) {
                                     minor_distance = sq_sum;
                                     nearest_centroid = c;
                                 }
                             }

                             arrayPclusters[point_id] = nearest_centroid;
                         }
                     }
                 });
         }).wait();
    }
}
