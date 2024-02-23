// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
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

template <typename FpTy>
void getNeighborhood(size_t n,
                     size_t dim,
                     FpTy *data,
                     size_t nq,
                     FpTy *query,
                     FpTy eps,
                     size_t *indices,
                     size_t *sizes)
{
    FpTy eps2 = eps * eps;
    size_t blockSize = 256;
    size_t nBlocks = n / blockSize + (n % blockSize > 0);
    for (size_t block = 0; block < nBlocks; block++) {
        size_t j1 = block * blockSize;
        size_t j2 = (block + 1 == nBlocks ? n : j1 + blockSize);

        for (size_t i = 0; i < nq; i++) {
            for (size_t j = j1; j < j2; j++) {
                FpTy dist = distance2(data + j * dim, query + i * dim, dim);
                if (dist <= eps2) {
                    size_t sz = sizes[i];
                    indices[i * n + sz] = j;
                    sizes[i]++;
                }
            }
        }
    }
}

template <typename FpTy> class DBScanKernel;

template <typename FpTy>
size_t dbscan_impl(queue q,
                   size_t n_samples,
                   size_t n_features,
                   FpTy *data,
                   FpTy eps,
                   size_t min_pts)
{
    size_t *sizes = new size_t[n_samples];
    memset(sizes, 0, n_samples * sizeof(size_t));

    size_t *d_indices =
        (size_t *)malloc_device(n_samples * n_samples * sizeof(size_t), q);
    size_t *d_sizes = (size_t *)malloc_device(n_samples * sizeof(size_t), q);

    // transfer data from host to device
    q.memcpy(d_sizes, sizes, n_samples * sizeof(size_t));
    q.wait();

    auto e = q.submit([&](handler &h) {
        h.parallel_for<DBScanKernel<FpTy>>(
            range<1>{n_samples}, [=](id<1> myID) {
                size_t i1 = myID[0];
                size_t i2 = (i1 + 1 == n_samples ? n_samples : i1 + 1);
                getNeighborhood<FpTy>(n_samples, n_features, data, i2 - i1,
                                      data + i1 * n_features, eps,
                                      d_indices + i1 * n_samples, d_sizes + i1);
            });
    });

    e.wait();

    // copy result back to host
    size_t *indices = new size_t[n_samples * n_samples];
    q.memcpy(indices, d_indices, n_samples * n_samples * sizeof(size_t));
    q.memcpy(sizes, d_sizes, n_samples * sizeof(size_t));

    q.wait();

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

    return nClusters;
}
