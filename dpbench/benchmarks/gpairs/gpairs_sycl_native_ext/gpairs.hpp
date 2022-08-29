/*
 * G-pairs algorithm:
 *   Input: n 3D points, given as x, y, and z vectors
 *          and associate vector of weights w
 *          nbins distance squared thresholds, as rbins vector
 *   Output:
 *          Vector of size nbins with computing
 *            hist[k] == sum( w[i1] * w[i2] *
 *                           bool( dist_sq(i1, i2) <= rbins[k] )
 *                       , [ 0<=i1<n, 0<=i2<n ])
 */

/*
 *  Description of the algorithm implementation
 *
 *  Each work-item processes n_wi points, work-group size is lws.
 *      n_groups = ceiling_quotient(n, n_wi * lws)
 *  Each group has local hist result accumulated over its local chunk
 *  of points.
 *
 *  Group histograms are added together as a post-processing step.
 */

#include <CL/sycl.hpp>

template <typename T>
inline T ceiling_quotient(const T n, const T m)
{
    return (n + m - 1) / m;
}

class eff_gpairs_kernel;

template <typename distT, typename wT>
sycl::event gpairs_usm(
    sycl::queue &q,
    size_t n,
    distT *x0, distT *y0, distT *z0, wT *w0,
    distT *x1, distT *y1, distT *z1, wT *w1,
    size_t nbins, distT *rbins, wT *hist,
    const std::vector<sycl::event> &depends = {})
{
    constexpr int n_wi = 16;
    constexpr int lws = 16;
    const size_t m = static_cast<size_t>(n_wi) * static_cast<size_t>(lws); //16*16=256
    const size_t n_groups = ceiling_quotient(n, m); //(16384+256-1)/256=64

    constexpr int local_hist_size = 8;
    sycl::event partial_hists_ev =
        q.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(depends);

                auto gwsRange = sycl::range<2>(n_groups * lws, n_groups * lws); //64*16,64*16 = 1024,1024
                auto lwsRange = sycl::range<2>(lws, lws); //16,16
                auto ndRange = sycl::nd_range<2>(gwsRange, lwsRange);

                const size_t slm_hist_size =
                    ceiling_quotient(nbins, size_t(local_hist_size)) *
		size_t(local_hist_size); // ((9+8-1)/8)*8=16

                /* shared local memory copy of histogram for work-group work-items */
                sycl::accessor<wT, 1, sycl::access::mode::read_write, sycl::access::target::local>
                    slm_hist(sycl::range<1>(slm_hist_size), cgh);

                cgh.parallel_for<class eff_gpairs_kernel>(
                    ndRange,
                    [=](sycl::nd_item<2> it)
                    {
                        const size_t lid0 = it.get_local_id(0);
                        const size_t gr0 = it.get_group(0); /* gid == gr * lws + lid */
                        const size_t lid1 = it.get_local_id(1);
                        const size_t gr1 = it.get_group(1); /* gid == gr * lws + lid */

                        /* initialize slm_hist with zeros */
                        for (size_t k = it.get_local_linear_id(); k < slm_hist_size; k += lws * lws)
                        {
                            slm_hist[k] = wT(0);
                        };

                        distT dsq_mat[n_wi * n_wi];
                        wT pw_mat[n_wi * n_wi];

                        const size_t offset0 = gr0 * n_wi * lws + lid0;
                        const size_t offset1 = gr1 * n_wi * lws + lid1;
                        /* work item works on pointer
                              j0 = gr0 * n_wi * lws + i0 * lws + lid0, and
                              j1 = gr1 * n_wi * lws + i1 * lws + lid1
                        */
                        {
                            /* compute (n_wi, n_wi) matrix of squared distances in work-item */
                            size_t j0 = offset0;
                            for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws)
                            {
                                const distT x0v = x0[j0], y0v = y0[j0], z0v = z0[j0];
                                const wT w0v = w0[j0];

                                size_t j1 = offset1;
                                for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws)
                                {
                                    pw_mat[i0 * n_wi + i1] = w0v * w1[j1];

                                    const distT dx(x0v - x1[j1]);
                                    const distT dy(y0v - y1[j1]);
                                    const distT dz(z0v - z1[j1]);

                                    dsq_mat[i0 * n_wi + i1] =
                                        dx * dx + dy * dy + dz * dz;
                                }
                            }
                        }
                        /* update slm_hist. Use work-item private buffer of 16 wT elements */
                        for (size_t k = 0; k < slm_hist_size; k += local_hist_size)
                        {
                            wT local_hist[local_hist_size];
                            // initialize local hist chunk to zero
                            for (int p = 0; p < local_hist_size; ++p)
                            {
                                local_hist[p] = wT(0);
                            }
                            // update local hist chunk using work-item's local dsq_mat
                            size_t j0 = offset0;
                            for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws)
                            {
                                size_t j1 = offset1;
                                for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws)
                                {
                                    distT dsq = dsq_mat[i0 * n_wi + i1];
                                    wT pw = pw_mat[i0 * n_wi + i1];
                                    wT zero(0);

                                    size_t pk = k;
                                    for (int p = 0; p < local_hist_size; ++p, ++pk)
                                    {
                                        local_hist[p] += (pk < nbins && dsq <= rbins[pk]) ? pw : zero;
                                    }
                                }
                            }
                            {
                                size_t pk = k;
                                for (int p = 0; p < local_hist_size; ++p, ++pk)
                                {
                                    auto v =
				      sycl::atomic_ref<
                                            wT, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::local_space>(
                                            slm_hist[pk]);
                                    v += local_hist[p];
                                }
                            }
                        }
                        /* write slm_hist into group_hists */
                        it.barrier(sycl::access::fence_space::local_space);
                        for (size_t k = it.get_local_linear_id(); k < nbins; k += lws * lws)
                        {
                            auto v =
                                sycl::atomic_ref<
				  wT, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    hist[k]);
                            v += slm_hist[k];
                        }
                    });
            });

    return partial_hists_ev;
}

/*
   for(gid=0; gid < n; gid += n_wi) {
       lid = gid % lws;
       gr = gid / lws;
       for(k=0; k < n_wi; ++k) {
           v = (gid + k < x_size) ? x[gr * n_wi * lws + k * lws + lid] : 0;
       }
   }
*/

template <typename distT, typename wT>
void gpairs_host_data_naive(
    size_t n,
    distT *x0, distT *y0, distT *z0, wT *w0,
    distT *x1, distT *y1, distT *z1, wT *w1,
    size_t nbins, distT *rbins, wT *hist)
{
    for (size_t i = 0; i < nbins; ++i)
    {
        hist[i] = wT(0);
    }
    for (size_t global_id0 = 0; global_id0 < n; ++global_id0)
    {
        for (size_t global_id1 = 0; global_id1 < n; ++global_id1)
        {
            distT dx(x0[global_id0] - x1[global_id1]);
            distT dy(y0[global_id0] - y1[global_id1]);
            distT dz(z0[global_id0] - z1[global_id1]);
            distT dsq = dx * dx + dy * dy + dz * dz;

            wT pw = w0[global_id0] * w1[global_id1];
            wT zero(0);
            for (size_t k = 0; k < nbins; ++k)
            {
                hist[k] += ((dsq <= rbins[k]) ? pw : zero);
            }
        }
    }
}

template <typename distT, typename wT>
void gpairs_host_data_chunked(
    size_t n,
    distT *x0, distT *y0, distT *z0, wT *w0,
    distT *x1, distT *y1, distT *z1, wT *w1,
    size_t nbins, distT *rbins, wT *hist)
{
    constexpr int n_wi = 8;
    constexpr int lws = 16;
    const size_t m = n_wi * lws;

    size_t n_groups = ceiling_quotient(n, m);

    for (size_t i = 0; i < nbins; ++i)
    {
        hist[i] = wT(0);
    }

    for (size_t gid0 = 0; gid0 < n_groups * lws; ++gid0)
    {
        size_t lid0 = gid0 % lws;
        size_t gr0 = gid0 / lws;
        for (size_t gid1 = 0; gid1 < n_groups * lws; ++gid1)
        {
            size_t lid1 = gid1 % lws;
            size_t gr1 = gid1 / lws;

            for (int i0 = 0; i0 < n_wi; ++i0)
            {
                for (int i1 = 0; i1 < n_wi; ++i1)
                {
                    size_t global_id0 = gr0 * n_wi * lws + i0 * lws + lid0;
                    size_t global_id1 = gr1 * n_wi * lws + i1 * lws + lid1;

                    if (global_id0 < n && global_id1 < n)
                    {
                        distT dx(x0[global_id0] - x1[global_id1]);
                        distT dy(y0[global_id0] - y1[global_id1]);
                        distT dz(z0[global_id0] - z1[global_id1]);
                        distT dsq = dx * dx + dy * dy + dz * dz;

                        wT dw = w0[global_id0] * w1[global_id1];
                        wT zero(0);
                        for (size_t k = 0; k < nbins; ++k)
                        {
                            hist[k] += ((dsq <= rbins[k]) ? dw : zero);
                        }
                    }
                }
            }
        }
    }
}

template <typename distT, typename wT>
void gpairs_host_data_chunked2(
    size_t n,
    distT *x0, distT *y0, distT *z0, wT *w0,
    distT *x1, distT *y1, distT *z1, wT *w1,
    size_t nbins, distT *rbins, wT *hist)
{
    constexpr int n_wi = 2;
    constexpr int lws = 16;
    const size_t m = n_wi * lws;

    size_t n_groups = ceiling_quotient(n, m);

    for (size_t gid0 = 0; gid0 < n_groups * lws; ++gid0)
    {
        size_t lid0 = gid0 % lws;
        size_t gr0 = gid0 / lws;
        for (size_t gid1 = 0; gid1 < n_groups * lws; ++gid1)
        {
            size_t lid1 = gid1 % lws;
            size_t gr1 = gid1 / lws;

            distT dsq_mat[n_wi * n_wi];
            wT pw_mat[n_wi * n_wi];

            size_t offset0 = gr0 * n_wi * lws + lid0;
            size_t offset1 = gr1 * n_wi * lws + lid1;

            size_t j0 = offset0;
            for (int i0 = 0; i0 < n_wi && j0 < n; ++i0, j0 += lws)
            {
                size_t j1 = offset1;
                const distT x0v = x0[j0], y0v = y0[j0], z0v = z0[j0];
                const wT w0v = w0[j0];
                for (int i1 = 0; i1 < n_wi && j1 < n; ++i1, j1 += lws)
                {
                    distT dx(x0v - x1[j1]);
                    distT dy(y0v - y1[j1]);
                    distT dz(z0v - z1[j1]);
                    distT dsq = dx * dx + dy * dy + dz * dz;

                    dsq_mat[i0 * n_wi + i1] = dsq;
                    pw_mat[i0 * n_wi + i1] = w0v * w1[j1];
                }
            }

            for (size_t k = 0; k < ceiling_quotient(nbins, size_t(16)) * 16; k += 16)
            {
                wT local_hist[16];

                for (int p = 0; p < 16; ++p)
                {
                    local_hist[p] = wT(0);
                }
                size_t j0 = offset0;
                for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws)
                {
                    size_t j1 = offset1;
                    for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws)
                    {
                        size_t pk = k;
                        distT dsq = dsq_mat[i0 * n_wi + i1];
                        wT pw = pw_mat[i0 * n_wi + i1];
                        wT zero(0);
                        for (int p = 0; p < 16; ++p, ++pk)
                        {
                            local_hist[p] += (pk < nbins && dsq <= rbins[pk]) ? pw : zero;
                        }
                    }
                }
                {
                    size_t pk = k;
                    for (int p = 0; (p < 16) && (pk < nbins); ++p, ++pk)
                    {
                        hist[pk] += local_hist[p];
                    }
                }
            }
        }
    }
}

#if 0
class dpbench_kernel;

template <typename distT, typename wT>
sycl::event gpairs_dpbench(
    sycl::queue &q,
    size_t n,
    distT *x1, distT *y1, distT *z1, wT *w1,
    distT *x2, distT *y2, distT *z2, wT *w2,
    size_t nbins, distT *rbins, wT *hist,
    const std::vector<sycl::event> &depends = {})
{
    sycl::event ev =
        q.submit(
            [&](sycl::handler &h)
            {
                auto range = sycl::range<1>{n};
                h.depends_on(depends);
                h.parallel_for<class dpbench_kernel>(
                    range,
                    [=](sycl::id<1> myID)
                    {
                        size_t i = myID[0];
                        distT px = x1[i];
                        distT py = y1[i];
                        distT pz = z1[i];
                        wT pw = w1[i];
                        for (size_t j = 0; j < n; j++)
                        {
                            distT qx = x2[j];
                            distT qy = y2[j];
                            distT qz = z2[j];
                            wT qw = w2[j];
                            distT dx = px - qx;
                            distT dy = py - qy;
                            distT dz = pz - qz;
                            wT wprod = pw * qw;
                            distT dsq = dx * dx + dy * dy + dz * dz;

                            int k = nbins - 1;
                            while (dsq <= rbins[k])
                            {
                                sycl::atomic_ref<
                                    wT, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    atomic_data(hist[k]);

                                atomic_data += wprod;
                                k = k - 1;
                                if (k < 0)
                                    break;
                            }
                        }
                    });
            });

    return ev;
}
#endif
