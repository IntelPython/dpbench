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

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <CL/sycl.hpp>

#include <dbscan.hpp>
#include <rdtsc.h>

using namespace std;
using namespace cl::sycl;

template <typename T>
void ReadInputFromBinFile(char const *filename,
                          char *data_ptr,
                          size_t data_size)
{
    ifstream file;
    file.open(filename, ios::in | ios::binary);
    if (file) {
        file.read(data_ptr, data_size * sizeof(T));
        file.close();
    }
    else {
        std::cout << "Input file - " << filename << " not found.\n";
        exit(0);
    }
}

double *readData(int nPoints, int nFeatures)
{
    int arrSize = nPoints * nFeatures;
    double *data = new double[arrSize];

    ReadInputFromBinFile<double>("X.bin", reinterpret_cast<char *>(data),
                                 arrSize);

    return data;
}

int main(int argc, char *argv[])
{
    size_t nPoints = 16384;
    size_t nFeatures = 10;
    size_t minPts = 20;
    double eps = 0.6;

    queue q;

    double *data = readData(nPoints, nFeatures);

    double *d_data =
        (double *)malloc_device(nPoints * nFeatures * sizeof(double), q);

    q.memcpy(d_data, data, nPoints * nFeatures * sizeof(double));

    q.wait();

    delete[] data;

    /* Warm up cycle */
    size_t nClusters =
        dbscan_impl<double>(q, nPoints, nFeatures, d_data, eps, minPts, false);

    nClusters =
        dbscan_impl<double>(q, nPoints, nFeatures, d_data, eps, minPts, true);

    return 0;
}
