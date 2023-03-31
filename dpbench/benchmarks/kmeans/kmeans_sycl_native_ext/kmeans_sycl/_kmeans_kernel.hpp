// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>

template <typename FpTy>
void kmeans_impl(sycl::queue q,
                 FpTy *arrayP,
                 size_t *arrayPclusters,
                 FpTy *arrayC,
                 FpTy *arrayCsum,
                 size_t *arrayCnumpoint,
                 size_t niters,
                 size_t npoints,
                 size_t ndims,
                 size_t ncentroids)
{
    q.submit([&](sycl::handler &h) {
         h.parallel_for<class theKernel_km>(
             sycl::range<1>{ncentroids}, [=](sycl::id<1> myID) {
                 size_t i = myID[0];

                 for (size_t j = 0; j < ndims; j++) {
                     arrayC[i * ndims + j] = arrayP[i * ndims + j];
                 }
             });
     }).wait();

    for (size_t i = 0; i < niters; i++) {
        // Group clusters based on distance
        q.submit([&](sycl::handler &h) {
            h.parallel_for<class theKernel>(
                sycl::range<1>{npoints}, [=](sycl::id<1> myID) {
                    size_t i0 = myID[0];

                    FpTy minor_distance = -1.0;
                    for (size_t i1 = 0; i1 < ncentroids; i1++) {
                        FpTy sq_sum = 0.0;
                        for (size_t j = 0; j < ndims; j++) {
                            FpTy dist =
                                arrayP[i0 * ndims + j] - arrayC[i1 * ndims + j];
                            sq_sum += dist * dist;
                        }
                        FpTy total_distance = cl::sycl::sqrt(sq_sum);
                        if (minor_distance > total_distance ||
                            minor_distance == -1.0)
                        {
                            minor_distance = total_distance;
                            arrayPclusters[i0] = i1;
                        }
                    }
                });
        });

        // Initialize Centroid sum
        q.submit([&](sycl::handler &h) {
             h.parallel_for<class theKernel_1>(
                 sycl::range<1>{ncentroids}, [=](sycl::id<1> myID_k1) {
                     size_t i0 = myID_k1[0];
                     for (size_t j = 0; j < ndims; j++) {
                         arrayCsum[i0 * ndims + j] = 0.0;
                     }
                     arrayCnumpoint[i0] = 0.0;
                 });
         }).wait();

        // Compute centroid sum
        q.submit([&](sycl::handler &h) {
             h.parallel_for<class theKernel_2>(
                 sycl::range<1>{npoints}, [=](sycl::id<1> myID) {
                     size_t i0 = myID[0];
                     size_t ci = arrayPclusters[i0];

                     for (size_t j = 0; j < ndims; j++) {
                         sycl::atomic_ref<
                             FpTy, sycl::memory_order::relaxed,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                             centroid_sum(arrayCsum[ci * ndims + j]);
                         centroid_sum += arrayP[i0 * ndims + j];
                     }

                     sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                                      sycl::memory_scope::system,
                                      sycl::access::address_space::global_space>
                         centroid_num_points(arrayCnumpoint[ci]);
                     centroid_num_points += 1;
                 });
         }).wait();

        // Update centroids
        q.submit([&](sycl::handler &h) {
             h.parallel_for<class theKernel_uc>(
                 sycl::range<1>{ncentroids}, [=](sycl::id<1> myID) {
                     size_t i0 = myID[0];
                     if (arrayCnumpoint[i0] > 0) {
                         for (size_t j = 0; j < ndims; j++) {
                             arrayC[i0 * ndims + j] =
                                 arrayCsum[i0 * ndims + j] / arrayCnumpoint[i0];
                         }
                     }
                 });
         }).wait();
    }
}
