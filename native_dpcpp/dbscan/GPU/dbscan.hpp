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

#include <cstdio>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <cstring>
#include <math.h>

#include <chrono>
#include <common.hpp>

#include <CL/sycl.hpp>

using namespace cl::sycl;

using namespace std;

double distance2 (double *a, double *b, size_t dim) {
    double dist = 0.0;
    for (size_t j = 0; j < dim; j++) {
        double diff = a[j] - b[j];
        dist += diff * diff;
    }
    return dist;
}

void getNeighborhood(size_t n, size_t dim, double* data, size_t nq, double* query, double eps, size_t* indices, size_t* sizes) { //, double* distances
    double eps2 = eps * eps;
    size_t blockSize = 256;
    size_t nBlocks = n / blockSize + (n % blockSize > 0);
    for (size_t block = 0; block < nBlocks; block++) {
        size_t j1 = block * blockSize;
        size_t j2 = (block + 1 == nBlocks ? n : j1 + blockSize);

        for (size_t i = 0; i < nq; i++) {
            for (size_t j = j1; j < j2; j++) {
                double dist = distance2(data + j * dim, query + i * dim, dim);
                if (dist <= eps2) {
                    size_t sz = sizes[i];
                    indices[i * n + sz] = j;
                    //distances[i * n + sz] = dist;
                    sizes[i]++;
                }
            }
        }
    }
}

size_t dbscan_reference_no_mem_save(queue* q, size_t n, size_t dim, double *data, double eps, size_t minPts, size_t *assignments) {
    size_t n2 = n * n;
    size_t* indices = new size_t[n2];
    //double* distances = new double[n2];
    size_t* sizes = new size_t[n];
    memset (sizes,0,n*sizeof(size_t));

    size_t blockSize = 1;
    size_t nBlocks = n / blockSize + (n % blockSize > 0);

    // transfer data from host to device
    double* d_data = (double*)malloc_shared( n*dim * sizeof(double), *q);
    size_t* d_indices = (size_t*)malloc_shared( n2 * sizeof(size_t), *q);
    size_t* d_sizes = (size_t*)malloc_shared( n * sizeof(size_t), *q);
    size_t* d_assignments = (size_t*)malloc_shared( n * sizeof(size_t), *q);

    q->memcpy(d_data, data, n * dim * sizeof(double));
    q->memcpy(d_sizes, sizes, n * sizeof(size_t));

    q->wait();

    // #pragma omp target teams distribute parallel for simd map(to: data[0:n*dim]) map(from: assignments[0:n], indices[0:n2]) map(tofrom: sizes[0:n]) //thread_limit(16) //, distances[0:n2]
    q->submit([&](handler& h) {
	h.parallel_for<class theKernel>(range<1>{nBlocks}, [=](id<1> myID) {
	    size_t block = myID[0];
	    size_t i1 = block * blockSize;
	    size_t i2 = (block + 1 == nBlocks ? n : i1 + blockSize);
	    for (size_t i = i1; i < i2; i++) {
	      d_assignments[i] = UNDEFINED;
	    }
	    getNeighborhood(n, dim, d_data, i2 - i1, d_data + i1 * dim, eps, d_indices + i1 * n, d_sizes + i1); //, distances + i1 * n
	  });
      });

    q->wait();

    // copy result back to host
    q->memcpy(indices, d_indices, n2 * sizeof(size_t));
    q->memcpy(assignments, d_assignments, n * sizeof(size_t));
    q->memcpy(sizes, d_sizes, n * sizeof(size_t));

    q->wait();

    size_t nClusters = 0;
    size_t nNoise = 0;
    for (size_t i = 0; i < n; i++) {
        if (assignments[i] != UNDEFINED) continue;
        size_t size = sizes[i];
        if (size < minPts) {
            nNoise++;
            assignments[i] = NOISE;
            continue;
        }
        nClusters++;
        assignments[i] = nClusters - 1;
        Queue qu;
        for (size_t j = 0; j < size; j++) {
            size_t nextPoint = indices[i * n + j];
            if (assignments[nextPoint] == NOISE) {
                nNoise--;
                assignments[nextPoint] = nClusters - 1;
            }
            else if (assignments[nextPoint] == UNDEFINED) {
                assignments[nextPoint] = nClusters - 1;
                qu.push(nextPoint);
            }
        }
        while (!qu.empty ()) {
            size_t curPoint = qu.pop();

            size_t size = sizes[curPoint];
            assignments[curPoint] = nClusters - 1;
            if (size < minPts) continue;

            for (size_t j = 0; j < size; j++) {
                size_t nextPoint = indices[curPoint * n + j];
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

    delete [] indices;
    //delete [] distances;
    delete [] sizes;

    return nClusters;
}
