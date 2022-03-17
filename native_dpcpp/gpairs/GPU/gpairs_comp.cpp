/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
0 *
 * SPDX-License-Identifier: MIT
 */

#include <CL/sycl.hpp>
#include "euro_opt.h"

using namespace cl::sycl;

template <typename T>
inline T ceiling_quotient(const T n, const T m)
{
    return (n + m - 1) / m;
}

//template <typename tfloat, typename tfloat, int n_wi=20, int private_hist_size=16, int lws0=8, int lws1=8>
sycl::event gpairs_slm(
    sycl::queue *q,
    size_t n,
    tfloat *x0, tfloat *y0, tfloat *z0, tfloat *w0,
    tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
    size_t nbins, tfloat *rbins, tfloat *hist)
//const std::vector<sycl::event> &depends = {})
{
    const unsigned int n_wi=20, private_hist_size=16, lws0=8, lws1=8;
    const size_t m0 = static_cast<size_t>(n_wi) * static_cast<size_t>(lws0);
    const size_t m1 = static_cast<size_t>(n_wi) * static_cast<size_t>(lws1);
    const size_t n_groups0 = ceiling_quotient(n, m0);
    const size_t n_groups1 = ceiling_quotient(n, m1);

    tfloat *dsq_mat = sycl::malloc_device<tfloat>(lws0 * lws1 * n_wi * n_wi, *q);

    sycl::event partial_hists_ev =
        q->submit(
            [&](sycl::handler &cgh)
            {
	      //cgh.depends_on(depends);

                auto gwsRange = sycl::range<2>(n_groups0 * lws0, n_groups1 * lws1);
                auto lwsRange = sycl::range<2>(lws0, lws1);
                auto ndRange = sycl::nd_range<2>(gwsRange, lwsRange);

                const size_t slm_hist_size =
                    ceiling_quotient(nbins, size_t(private_hist_size)) *
                    size_t(private_hist_size);

                /* shared local memory copy of histogram for work-group work-items */
                sycl::accessor<tfloat, 1, sycl::access::mode::read_write, sycl::access::target::local>
                    slm_hist(sycl::range<1>(slm_hist_size), cgh);

                cgh.parallel_for<class eff_gpairs_kernel_usm>(
                    ndRange,
                    [=](sycl::nd_item<2> it)
                    {
                        const size_t lid0 = it.get_local_id(0);
                        const size_t gr0 = it.get_group(0); /* gid0 == gr0 * lws0 + lid0 */
                        const size_t lid1 = it.get_local_id(1);
                        const size_t gr1 = it.get_group(1); /* gid1 == gr1 * lws1 + lid1 */

                        /* initialize slm_hist with zeros */
                        for (size_t k = it.get_local_linear_id(); k < slm_hist_size; k += lws0 * lws1)
                        {
                            slm_hist[k] = tfloat(0);
                        };
                        it.barrier(sycl::access::fence_space::local_space);

                        // tfloat dsq_mat[n_wi * n_wi];
                        // tfloat w0_vec[n_wi];
                        // tfloat w1_vec[n_wi];

                        const size_t offset0 = gr0 * n_wi * lws0 + lid0;
                        const size_t offset1 = gr1 * n_wi * lws1 + lid1;
                        /* work item works on pointer
                              j0 = gr0 * n_wi * lws0 + i0 * lws0 + lid0, and
                              j1 = gr1 * n_wi * lws1 + i1 * lws1 + lid1
                        */
                        {
                            // size_t j1 = offset1;
                            // for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                            // {
                            //     w1_vec[i1] = w1[j1];
                            // }

                            /* compute (n_wi, n_wi) matrix of squared distances in work-item */
                            size_t j0 = offset0;
                            for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws0)
                            {
                                const tfloat x0v = x0[j0], y0v = y0[j0], z0v = z0[j0];
                                //w0_vec[i0] = w0[j0];

                                size_t j1 = offset1;
                                for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                                {
                                    const tfloat dx(x0v - x1[j1]);
                                    const tfloat dy(y0v - y1[j1]);
                                    const tfloat dz(z0v - z1[j1]);

				    size_t mat_id = (it.get_local_linear_id() * n_wi + i0) * n_wi + i1;
                                    dsq_mat[mat_id] =
                                        dx * dx + dy * dy + dz * dz;
                                }
                            }
                        }
			// needed since previous loop writes out into global memory dsq_mat
                        // allocated for each work-item in work-group
                        it.barrier(sycl::access::fence_space::local_space);
			
                        /* update slm_hist. Use work-item private buffer of 16 tfloat elements */
                        for (size_t k = 0; k < slm_hist_size; k += private_hist_size)
                        {
                            tfloat private_hist[private_hist_size];
                            // initialize local hist chunk to zero
                            for (int p = 0; p < private_hist_size; ++p)
                            {
                                private_hist[p] = tfloat(0);
                            }
                            // update local hist chunk using work-item's local dsq_mat
                            size_t j0 = offset0;
                            for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws0)
                            {
                                size_t j1 = offset1;
                                for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                                {
				  size_t mat_id = (it.get_local_linear_id() * n_wi + i0) * n_wi + i1;
				  tfloat dsq = dsq_mat[mat_id/*i0 * n_wi + i1*/];
				  tfloat pw = w0[j0] * w1[j1];
				  tfloat zero(0);

				  size_t pk = k;
#pragma unroll
				  for (int p = 0; p < private_hist_size; ++p, ++pk)
                                    {
				      private_hist[p] += (pk < nbins && dsq <= rbins[pk]) ? pw : zero;
                                    }
                                }
                            }
                            {
			      size_t pk = k;
			      for (int p = 0; p < private_hist_size; ++p, ++pk)
                                {
			    	  auto v =
			    	    sycl::atomic_ref<
			    	      tfloat, sycl::memory_order::relaxed,
			    	    sycl::memory_scope::device,
			    	    sycl::access::address_space::local_space>(
			    						      slm_hist[pk]);
			    	  v += private_hist[p];
                                }
                            }
                        }
                        /* write slm_hist into group_hists */
                        it.barrier(sycl::access::fence_space::local_space);
                        for (size_t k = it.get_local_linear_id(); k < nbins; k += lws0 * lws1)
                        {
                            auto v =
                                sycl::atomic_ref<
                                    tfloat, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    hist[k]);
                            v += slm_hist[k];
                        }
                    });
            });
				  
    partial_hists_ev = q->submit([&](sycl::handler &cgh) {
    	cgh.depends_on(partial_hists_ev);
    	sycl::context ctx = q->get_context();
    	cgh.host_task([=]() {sycl::free(dsq_mat, ctx);});
      });
    
    return partial_hists_ev;
}

//template <typename tfloat, typename tfloat, int n_wi=20, int private_hist_size=16, int lws0=8, int lws1=8>
sycl::event gpairs_no_slm(
    sycl::queue *q,
    size_t n,
    tfloat *x0, tfloat *y0, tfloat *z0, tfloat *w0,
    tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
    size_t nbins, tfloat *rbins, tfloat *hist)
//const std::vector<sycl::event> &depends = {})
{

    const unsigned int n_wi=20, private_hist_size=16, lws0=16, lws1=16;
    const size_t m0 = static_cast<size_t>(n_wi) * static_cast<size_t>(lws0);
    const size_t m1 = static_cast<size_t>(n_wi) * static_cast<size_t>(lws1);
    const size_t n_groups0 = ceiling_quotient(n, m0);
    const size_t n_groups1 = ceiling_quotient(n, m1);

    sycl::event partial_hists_ev =
        q->submit(
            [&](sycl::handler &cgh)
            {
	      //cgh.depends_on(depends);

                auto gwsRange = sycl::range<2>(n_groups0 * lws0, n_groups1 * lws1);
                auto lwsRange = sycl::range<2>(lws0, lws1);
                auto ndRange = sycl::nd_range<2>(gwsRange, lwsRange);

                const size_t slm_hist_size =
                    ceiling_quotient(nbins, size_t(private_hist_size)) *
                    size_t(private_hist_size);

                cgh.parallel_for<class eff_gpairs_kernel_no_slm>(
                    ndRange,
                    [=](sycl::nd_item<2> it)
                    {
                        const size_t lid0 = it.get_local_id(0);
                        const size_t gr0 = it.get_group(0); /* gid0 == gr0 * lws0 + lid0 */
                        const size_t lid1 = it.get_local_id(1);
                        const size_t gr1 = it.get_group(1); /* gid1 == gr1 * lws1 + lid1 */

                        tfloat dsq_mat[n_wi * n_wi];
                        tfloat w0_vec[n_wi];
                        tfloat w1_vec[n_wi];

                        const size_t offset0 = gr0 * n_wi * lws0 + lid0;
                        const size_t offset1 = gr1 * n_wi * lws1 + lid1;
                        /* work item works on pointer
                              j0 = gr0 * n_wi * lws0 + i0 * lws0 + lid0, and
                              j1 = gr1 * n_wi * lws1 + i1 * lws1 + lid1
                        */
                        {
                            size_t j1 = offset1;
                            for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                            {
                                w1_vec[i1] = w1[j1];
                            }

                            /* compute (n_wi, n_wi) matrix of squared distances in work-item */
                            size_t j0 = offset0;
                            for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws0)
                            {
                                const tfloat x0v = x0[j0], y0v = y0[j0], z0v = z0[j0];
                                w0_vec[i0] = w0[j0];

                                size_t j1 = offset1;
                                for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                                {
                                    const tfloat dx(x0v - x1[j1]);
                                    const tfloat dy(y0v - y1[j1]);
                                    const tfloat dz(z0v - z1[j1]);

                                    dsq_mat[i0 * n_wi + i1] =
                                        dx * dx + dy * dy + dz * dz;
                                }
                            }
                        }
                        /* update slm_hist. Use work-item private buffer of 16 tfloat elements */
                        for (size_t k = 0; k < slm_hist_size; k += private_hist_size)
                        {
                            tfloat private_hist[private_hist_size];
                            // initialize local hist chunk to zero
                            for (int p = 0; p < private_hist_size; ++p)
                            {
                                private_hist[p] = tfloat(0);
                            }
                            // update local hist chunk using work-item's local dsq_mat
                            size_t j0 = offset0;
                            for (int i0 = 0; (i0 < n_wi) && (j0 < n); ++i0, j0 += lws0)
                            {
                                size_t j1 = offset1;
                                for (int i1 = 0; (i1 < n_wi) && (j1 < n); ++i1, j1 += lws1)
                                {
                                    tfloat dsq = dsq_mat[i0 * n_wi + i1];
                                    tfloat pw = w0_vec[i0] * w1_vec[i1];
                                    tfloat zero(0);

                                    size_t pk = k;
                                    for (int p = 0; p < private_hist_size; ++p, ++pk)
                                    {
                                        private_hist[p] += (pk < nbins && dsq <= rbins[pk]) ? pw : zero;
                                    }
                                }
                            }
                            {
                                size_t pk = k;
                                for (int p = 0; p < private_hist_size; ++p, ++pk)
                                {
                                    auto v =
                                        sycl::atomic_ref<
                                            tfloat, sycl::memory_order::relaxed,
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

sycl::event gpairs_orig( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2, size_t nbins,tfloat* rbins,tfloat* results_test) {

  tfloat* t_results_tmp = (tfloat*)malloc((DEFAULT_NBINS) * sizeof(tfloat));
  memset (t_results_tmp,0,(DEFAULT_NBINS) * sizeof(tfloat));
  
  tfloat* results_tmp = (tfloat*)malloc_device( (DEFAULT_NBINS) * sizeof(tfloat), *q);
  q->memcpy(results_tmp, t_results_tmp, (DEFAULT_NBINS) * sizeof(tfloat));

  q->wait();

  q->submit([&](handler& h) {
      h.parallel_for<class theKernel>(range<1>{npoints}, [=](id<1> myID) {
  	  size_t i = myID[0];

  	  tfloat px = x1[i];
  	  tfloat py = y1[i];
  	  tfloat pz = z1[i];
  	  tfloat pw = w1[i];
  	  for (size_t j = 0; j < npoints; j++) {
  	    tfloat qx = x2[j];
  	    tfloat qy = y2[j];
  	    tfloat qz = z2[j];
  	    tfloat qw = w2[j];
  	    tfloat dx = px - qx;
  	    tfloat dy = py - qy;
  	    tfloat dz = pz - qz;
  	    tfloat wprod = pw * qw;
  	    tfloat dsq = dx*dx + dy*dy + dz*dz;

	    if (dsq <= rbins[nbins-1]) {
	      for (size_t k = nbins-1; k >= 0; k--) {
		if (k==0 || dsq > rbins[k-1]) {
		  sycl::atomic_ref<tfloat, sycl::memory_order::relaxed,
				   sycl::memory_scope::device,
				   sycl::access::address_space::global_space>atomic_data(results_tmp[k]);
		  atomic_data += wprod;
		  break;
		}
	      }
	    }
  	  }
  	});
    });

  q->wait();

  sycl::event partial_hists_ev = q->submit([&](handler& h) {
      h.parallel_for<class MergeKernel>(nbins, [=](id<1> myID) {
  	  int id = myID[0];
	  for (int j=0; j <= id; j++) {
	    results_test[id] += results_tmp[j];
	  }
  	});
    });

  return partial_hists_ev;
}

void gpairs_host( queue* q, size_t npoints, tfloat* x1, tfloat* y1, tfloat* z1, tfloat* w1, tfloat* x2,tfloat* y2,tfloat* z2, tfloat* w2, size_t nbins,tfloat* rbins,tfloat* results_test) {
  for (size_t i = 0; i < npoints; i++) {
    tfloat px = x1[i];
    tfloat py = y1[i];
    tfloat pz = z1[i];
    tfloat pw = w1[i];
    for (size_t j = 0; j < npoints; j++) {
      tfloat qx = x2[j];
      tfloat qy = y2[j];
      tfloat qz = z2[j];
      tfloat qw = w2[j];
      tfloat dx = px - qx;
      tfloat dy = py - qy;
      tfloat dz = pz - qz;
      tfloat wprod = pw * qw;
      tfloat dsq = dx*dx + dy*dy + dz*dz;

      size_t k = nbins - 1;
      while(dsq <= rbins[k]) {
	results_test[k-1] += wprod;
	k = k-1;
	if (k <=0) break;
      }
    }
  }
}
