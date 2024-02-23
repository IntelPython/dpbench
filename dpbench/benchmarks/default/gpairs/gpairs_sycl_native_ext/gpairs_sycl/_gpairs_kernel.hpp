// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>

template <typename T> inline T ceiling_quotient(const T n, const T m)
{
    return (n + m - 1) / m;
}

template <typename FpTy>
sycl::event gpairs_impl(sycl::queue q,
                        size_t n,
                        size_t nbins,
                        const FpTy *x0,
                        const FpTy *y0,
                        const FpTy *z0,
                        const FpTy *w0,
                        const FpTy *x1,
                        const FpTy *y1,
                        const FpTy *z1,
                        const FpTy *w1,
                        const FpTy *rbins,
                        FpTy *hist)
{

    const unsigned int n_wi = 32, private_hist_size = 32, lws0 = 16, lws1 = 16;
    const size_t m0 = static_cast<size_t>(n_wi) * static_cast<size_t>(lws0);
    const size_t m1 = static_cast<size_t>(n_wi) * static_cast<size_t>(lws1);
    const size_t n_groups0 = ceiling_quotient(n, m0);
    const size_t n_groups1 = ceiling_quotient(n, m1);

    sycl::event partial_hists_ev = q.submit([&](sycl::handler &cgh) {
        auto gwsRange = sycl::range<2>(n_groups0 * lws0, n_groups1 * lws1);
        auto lwsRange = sycl::range<2>(lws0, lws1);
        auto ndRange = sycl::nd_range<2>(gwsRange, lwsRange);

        const size_t slm_hist_size =
            ceiling_quotient(nbins, size_t(private_hist_size)) *
            size_t(private_hist_size);

        cgh.parallel_for(ndRange, [=](sycl::nd_item<2> it) {
            const size_t lid0 = it.get_local_id(0);
            const size_t gr0 = it.get_group(0); /* gid0 == gr0 * lws0 + lid0 */
            const size_t lid1 = it.get_local_id(1);
            const size_t gr1 = it.get_group(1); /* gid1 == gr1 * lws1 + lid1 */

            FpTy dsq_mat[n_wi * n_wi];
            FpTy w0_vec[n_wi];
            FpTy w1_vec[n_wi];

            const size_t offset0 = gr0 * n_wi * lws0 + lid0;
            const size_t offset1 = gr1 * n_wi * lws1 + lid1;
            /* work item works on pointer
                  j0 = gr0 * n_wi * lws0 + i0 * lws0 + lid0, and
                  j1 = gr1 * n_wi * lws1 + i1 * lws1 + lid1
            */
            {
                size_t j1 = offset1;
                for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1) {
                    w1_vec[i1] = w1[j1];
                }

                /* compute (n_wi, n_wi) matrix of squared distances in
                 * work-item */
                size_t j0 = offset0;
                for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws0) {
                    const FpTy x0v = x0[j0], y0v = y0[j0], z0v = z0[j0];
                    w0_vec[i0] = w0[j0];

                    size_t j1 = offset1;
                    for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                    {
                        const FpTy dx(x0v - x1[j1]);
                        const FpTy dy(y0v - y1[j1]);
                        const FpTy dz(z0v - z1[j1]);

                        dsq_mat[i0 * n_wi + i1] = dx * dx + dy * dy + dz * dz;
                    }
                }
            }
            /* update slm_hist. Use work-item private buffer of 16 FpTy
             * elements */
            for (size_t k = 0; k < slm_hist_size; k += private_hist_size) {
                FpTy private_hist[private_hist_size];
                // initialize local hist chunk to zero
                for (int p = 0; p < private_hist_size; ++p) {
                    private_hist[p] = FpTy(0);
                }
                // update local hist chunk using work-item's local dsq_mat
                size_t j0 = offset0;
                for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws0) {
                    size_t j1 = offset1;
                    for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                    {
                        FpTy dsq = dsq_mat[i0 * n_wi + i1];
                        FpTy pw = w0_vec[i0] * w1_vec[i1];
                        FpTy zero(0);

                        size_t pk = k;
                        for (int p = 0; p < private_hist_size; ++p, ++pk) {
                            private_hist[p] +=
                                (pk < nbins && dsq <= rbins[pk]) ? pw : zero;
                        }
                    }
                }
                {
                    size_t pk = k;
                    for (int p = 0; p < private_hist_size; ++p, ++pk) {
                        auto v = sycl::atomic_ref<
                            FpTy, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            hist[pk]);
                        v += private_hist[p];
                    }
                }
            }
        });
    });

    return partial_hists_ev;
}
