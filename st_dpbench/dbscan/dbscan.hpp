/*
Copyright (c) 2020, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <rdtsc.h>
#include <stdlib.h>

#define UNDEFINED -2
#define NOISE -1

using namespace sycl;

struct cluster_queue
{
    static const size_t defaultSize = 10;

    size_t *values;

    size_t capacity;
    size_t head;
    size_t tail;

    cluster_queue(size_t cap = defaultSize)
    {
        capacity = cap;
        head = tail = 0;
        values = new size_t[capacity];
    }

    ~cluster_queue() { delete[] values; }

    void resize(size_t newCapacity)
    {
        size_t *newValues = new size_t[newCapacity];

        memcpy(newValues, values, sizeof(size_t) * tail);

        delete[] values;

        capacity = newCapacity;
        values = newValues;
    }

    void push(size_t val)
    {
        if (tail == capacity) {
            resize(2 * capacity);
        }

        values[tail] = val;
        tail++;
    }

    inline size_t pop()
    {
        if (head < tail) {
            head++;
            return values[head - 1];
        }

        return -1;
    }

    inline bool empty() { return head == tail; }

    inline size_t getSize() { return tail - head; }
};

template <typename FpTy> FpTy distance2(FpTy *a, FpTy *b, size_t dim)
{
    FpTy dist = 0.0;
    for (size_t j = 0; j < dim; j++) {
        FpTy diff = a[j] - b[j];
        dist += diff * diff;
    }
    return dist;
}

template <typename FpTy> class DBScanKernel;

template <typename FpTy>
size_t dbscan_impl(queue &q,
                   size_t n_samples,
                   size_t n_features,
                   FpTy *data,
                   FpTy eps,
                   size_t min_pts,
                   bool measure_time)
{
    clock_t t1 = 0, t2 = 0;
    double time = 0.0;

    size_t *sizes = new size_t[n_samples];

    size_t *d_indices =
        (size_t *)malloc_device(n_samples * n_samples * sizeof(size_t), q);
    size_t *d_sizes = (size_t *)malloc_device(n_samples * sizeof(size_t), q);

    if (measure_time)
        t1 = timer_rdtsc();

    q.submit([&](handler &h) {
         h.parallel_for(range<1>{n_samples}, [=](id<1> myID) {
             auto i = myID[0];

             for (auto j = 0; j < n_samples; j++) {
                 double dist = 0.0;
                 for (auto m = 0; m < n_features; m++) {
                     auto diff =
                         data[i * n_features + m] - data[j * n_features + m];
                     dist += diff * diff;
                 }

                 if (dist <= eps) {
                     d_indices[i * n_samples + d_sizes[i]] = j;
                     d_sizes[i]++;
                 }
             }
         });
     }).wait();

    if (measure_time) {
        t2 = timer_rdtsc();
        time = ((double)(t2 - t1) / getHz());
        printf("Device Compute Time: %.6lf\n", time);
        t1 = timer_rdtsc();
    }

    // copy result back to host
    size_t *indices = new size_t[n_samples * n_samples];
    q.memcpy(indices, d_indices, n_samples * n_samples * sizeof(size_t));
    q.memcpy(sizes, d_sizes, n_samples * sizeof(size_t));

    q.wait();

    if (measure_time) {
        t2 = timer_rdtsc();
        time = ((double)(t2 - t1) / getHz());
        printf("Data Transfer Time: %.6lf\n", time);
        t1 = timer_rdtsc();
    }

    size_t *assignments = new size_t[n_samples];
    for (size_t i = 0; i < n_samples; i++) {
        assignments[i] = UNDEFINED;
    }

    size_t nClusters = 0;
    size_t nNoise = 0;
    for (size_t i = 0; i < n_samples; i++) {
        if (assignments[i] != UNDEFINED)
            continue;
        size_t size = sizes[i];
        if (size < min_pts) {
            nNoise++;
            assignments[i] = NOISE;
            continue;
        }
        nClusters++;
        assignments[i] = nClusters - 1;
        cluster_queue qu;
        for (size_t j = 0; j < size; j++) {
            size_t nextPoint = indices[i * n_samples + j];
            if (assignments[nextPoint] == NOISE) {
                nNoise--;
                assignments[nextPoint] = nClusters - 1;
            }
            else if (assignments[nextPoint] == UNDEFINED) {
                assignments[nextPoint] = nClusters - 1;
                qu.push(nextPoint);
            }
        }
        while (!qu.empty()) {
            size_t curPoint = qu.pop();

            size_t size = sizes[curPoint];
            assignments[curPoint] = nClusters - 1;
            if (size < min_pts)
                continue;

            for (size_t j = 0; j < size; j++) {
                size_t nextPoint = indices[curPoint * n_samples + j];
                if (assignments[nextPoint] == NOISE) {
                    nNoise--;
                    assignments[nextPoint] = nClusters - 1;
                }
                else if (assignments[nextPoint] == UNDEFINED) {
                    assignments[nextPoint] = nClusters - 1;
                    qu.push(nextPoint);
                }
            }
        }
    }

    delete[] indices;
    delete[] sizes;

    if (measure_time) {
        t2 = timer_rdtsc();
        time = ((double)(t2 - t1) / getHz());
        printf("Sequential host time: %.6lf\n", time);
    }

    return nClusters;
}
