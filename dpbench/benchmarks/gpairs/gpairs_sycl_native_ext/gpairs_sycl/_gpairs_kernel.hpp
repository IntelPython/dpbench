// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <algorithm>

template <typename T> inline T ceiling_quotient(const T n, const T m)
{
    return (n + m - 1) / m;
}

template <class NumT, class DenT> NumT DivUp(NumT numerator, DenT denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <class VT, class BT> VT Align(VT value, BT base)
{
    return base * DivUp(value, base);
}

struct local_atomic_add
{
    template <typename T1, typename T2>
    static inline void eqadd(T1 &lhs, T2 rhs)
    {
        sycl::atomic_ref<T1, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            _lhs(lhs);

        _lhs += rhs;
    }
};

struct local_atomic_cmp_exchng
{
    template <typename T1, typename T2>
    static inline void eqadd(T1 &lhs, T2 rhs)
    {
        sycl::atomic_ref<T1, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            _lhs(lhs);

        auto curr = _lhs.load(sycl::memory_order::relaxed);
        while (!_lhs.compare_exchange_strong(curr, T1(curr + rhs),
                                             sycl::memory_order::relaxed))
        {
        }
    }
};

template <typename FpTy, class LocalAtomicOp>
sycl::event gpairs_impl_(sycl::queue q,
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
    int max_worg_group_size = std::min(
        1024, int(q.get_device()
                      .get_info<sycl::info::device::max_work_group_size>()));

    sycl::event partial_hists_ev = q.submit([&](sycl::handler &cgh) {
        int dim1 = 64;

        auto local_size = sycl::range<2>(max_worg_group_size / dim1, dim1);
        int WPI = 64;
        auto global_size = sycl::range<2>(Align(DivUp(n, WPI), local_size[0]),
                                          Align(n, local_size[1]));

        auto work_size = sycl::nd_range<2>(global_size, local_size);

        int local_copies = 16;
        auto localBins =
            sycl::local_accessor<FpTy, 1>(sycl::range<1>(nbins), cgh);
        auto localHist = sycl::local_accessor<FpTy, 2>(
            sycl::range<2>(local_copies, nbins), cgh);
        cgh.parallel_for(work_size, [=](sycl::nd_item<2> item) {
            auto group = item.get_group();
            auto lid = item.get_local_linear_id();

            auto gid0_ = WPI * item.get_global_id(0);
            auto gid1 = item.get_global_id(1);

            for (int i = lid; i < nbins; i += max_worg_group_size)
                localBins[i] = rbins[i];

            for (int i = lid; i < nbins; i += max_worg_group_size)
                for (int j = 0; j < local_copies; ++j)
                    localHist[j][i] = 0;

            sycl::group_barrier(group, sycl::memory_scope::work_group);

            auto _x0 = x0[gid1];
            auto _y0 = y0[gid1];
            auto _z0 = z0[gid1];
            auto _w0 = w0[gid1];

            auto lastBin = localBins[nbins - 1];

            for (int i = 0; i < WPI; ++i) {
                auto gid0 = gid0_ + i;
                if (gid0 < n and gid1 < n) {
                    auto _x1 = x1[gid0];
                    auto _y1 = y1[gid0];
                    auto _z1 = z1[gid0];
                    auto _w1 = w1[gid0];

                    auto dist = (_x0 - _x1) * (_x0 - _x1) +
                                (_y0 - _y1) * (_y0 - _y1) +
                                (_z0 - _z1) * (_z0 - _z1);

                    int bin_id = 0;
                    if (dist < lastBin) {
                        bin_id = std::upper_bound(&localBins[0],
                                                  &localBins[nbins], dist) -
                                 &localBins[0];

                        LocalAtomicOp::eqadd(
                            localHist[lid % local_copies][bin_id], _w0 * _w1);
                    }
                }
            }

            sycl::group_barrier(group, sycl::memory_scope::work_group);

            for (int i = lid; i < nbins; i += max_worg_group_size) {
                sycl::atomic_ref<FpTy, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    _hist(hist[i]);
                auto hist_val = FpTy(0);
                for (int j = 0; j < local_copies; ++j)
                    hist_val += localHist[j][i];
                _hist += hist_val;
            }
        });
    });

    auto e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(partial_hists_ev);
        auto local_size = sycl::range<1>(max_worg_group_size);
        auto global_size = sycl::range<1>(Align(nbins, local_size[0]));

        auto work_size = sycl::nd_range<1>(local_size, local_size);

        auto localHist =
            sycl::local_accessor<FpTy, 1>(sycl::range<1>(nbins), cgh);
        auto res = sycl::local_accessor<FpTy, 1>(sycl::range<1>(nbins), cgh);
        cgh.parallel_for(work_size, [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            auto lid = item.get_local_linear_id();

            for (int i = lid; i < nbins; i += max_worg_group_size)
                localHist[i] = hist[i];

            sycl::group_barrier(group, sycl::memory_scope::work_group);
            sycl::joint_inclusive_scan(group, &localHist[0], &localHist[nbins],
                                       &res[0], sycl::plus<FpTy>());
            sycl::group_barrier(group, sycl::memory_scope::work_group);

            for (int i = lid; i < nbins; i += max_worg_group_size)
                hist[i] = res[i];
        });
    });

    return e;
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
    return gpairs_impl_<FpTy, local_atomic_add>(q, n, nbins, x0, y0, z0, w0, x1,
                                                y1, z1, w1, rbins, hist);
}

template <>
sycl::event gpairs_impl<double>(sycl::queue q,
                                size_t n,
                                size_t nbins,
                                const double *x0,
                                const double *y0,
                                const double *z0,
                                const double *w0,
                                const double *x1,
                                const double *y1,
                                const double *z1,
                                const double *w1,
                                const double *rbins,
                                double *hist)
{
    const sycl::device &d = q.get_device();

    if (d.is_gpu()) {
        return gpairs_impl_<double, local_atomic_cmp_exchng>(
            q, n, nbins, x0, y0, z0, w0, x1, y1, z1, w1, rbins, hist);
    }
    else {
        return gpairs_impl_<double, local_atomic_add>(
            q, n, nbins, x0, y0, z0, w0, x1, y1, z1, w1, rbins, hist);
    }
}
