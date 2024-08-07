/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "euro_opt.h"
#include "rdtsc.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/sycl.hpp>

template <typename T> inline T ceiling_quotient(const T n, const T m)
{
    return (n + m - 1) / m;
}

template <typename FpTy>
void gpairs_impl(sycl::queue &q,
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
                 FpTy *result_test)
{

    FpTy *result = (FpTy *)malloc_device(n * (nbins) * sizeof(FpTy), q);
    q.wait();

    q.submit([&](handler &h) {
        h.parallel_for(sycl::range{n}, [=](sycl::id<1> myID) {
            size_t i = myID[0];

            FpTy px = x0[i];
            FpTy py = y0[i];
            FpTy pz = z0[i];
            FpTy pw = w0[i];
            for (size_t j = 0; j < n; j++) {
                FpTy qx = x1[j];
                FpTy qy = y1[j];
                FpTy qz = z1[j];
                FpTy qw = w1[j];
                FpTy dx = px - qx;
                FpTy dy = py - qy;
                FpTy dz = pz - qz;
                FpTy wprod = pw * qw;
                FpTy dsq = dx * dx + dy * dy + dz * dz;

                if (dsq <= rbins[nbins - 1]) {
                    for (int k = nbins - 1; k >= 0; k--) {
                        if (dsq > rbins[k]) {
                            result[i * nbins + (k + 1)] += wprod;
                            break;
                        }
                        else if (k == 0)
                            result[i * nbins + k] += wprod;
                    }
                }
            }

            // Iterate through work-item private result from n-2->0(where
            // n=nbins-1). For each j'th bin add it's contents to all bins from
            // j+1 to n-1
            for (int j = nbins - 2; j >= 0; j--) {
                for (int k = j + 1; k < nbins; k++) {
                    result[i * nbins + k] += result[i * nbins + j];
                }
            }
        });
    });

    q.wait();

    q.submit([&](handler &h) {
        h.parallel_for(nbins, [=](id<1> myID) {
            int col_id = myID[0];

            for (size_t i = 1; i < n; i++) {
                result[col_id] += result[i * nbins + col_id];
            }
            result_test[col_id] = result[col_id];
        });
    });

    q.wait();
}

int main(int argc, char *argv[])
{
    size_t nopt = 524288;

    clock_t t1 = 0, t2 = 0;

    queue q;

    tfloat *x1, *y1, *z1, *w1, *x2, *y2, *z2, *w2, *rbins, *results_test;

    /* Allocate arrays, generate input data */
    InitData(q, nopt, &x1, &y1, &z1, &w1, &x2, &y2, &z2, &w2, &rbins,
             &results_test);

    /* Warm up cycle */
    gpairs_impl<double>(q, nopt, DEFAULT_NBINS, x1, y1, z1, w1, x2, y2, z2, w2,
                        rbins, results_test);

    ResetResult(q, results_test);

    t1 = timer_rdtsc();
    gpairs_impl<double>(q, nopt, DEFAULT_NBINS, x1, y1, z1, w1, x2, y2, z2, w2,
                        rbins, results_test);
    t2 = timer_rdtsc();

    printf("TIME: %.6lf\n", ((double)(t2 - t1) / getHz()));
    fflush(stdout);

    /* Deallocate arrays */
    FreeData(q, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_test);

    return 0;
}
