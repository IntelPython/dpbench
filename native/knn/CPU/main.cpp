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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <array>
#include "knn.h"
#include "rdtsc.h"
#include <sstream>


// std=c++11 or higher (c++14, c++17)
using dtype = double;


static int stoi(char* h) {
    std::stringstream in(h);
    int res;
    in >> res;
    return res;
}

static double stof(char* h) {
    std::stringstream in(h);
    double res;
    in >> res;
    return res;
}

static dtype rand32(dtype a, dtype b) {
    return abs((rand() << 16) | rand()) % 1000000000 / 1000000000.0 * (b - a) + a;
}


std::vector<std::array<dtype, dataDim>> gen_data_x(size_t data_size)
{
    std::vector<std::array<dtype, dataDim>> data;

    for (size_t i = 0; i < data_size; ++i)
    {
        std::array<dtype, dataDim> x = {};
        for (size_t j = 0; j < dataDim; ++j)
        {
            x[j] = rand32(0, 1);
        }
        data.push_back(x);
    }

    return data;
}


std::vector<size_t> gen_data_y(size_t data_size)
{
    std::vector<size_t> labels;
    for (size_t i = 0; i < data_size; ++i)
    {
        labels.push_back((size_t)rand() % classesNum);
    }

    return labels;
}

std::vector<dtype> gen_data_x_vector(size_t data_size)
{
    std::vector<dtype> data;

    for (size_t i = 0; i < data_size; ++i)
    {
        for (size_t j = 0; j < dataDim; ++j)
        {
            data.push_back(rand32(0, 1));
        }

    }

    return data;
}


int main(int argc, char* argv[]) {
    int STEPS = 10;

    size_t nPoints_train = pow(2, 10);
    size_t nPoints = pow(2, 10);

    size_t nFeatures = 1;
    size_t minPts = 5;
    double eps = 1.0;

    if (argc < 2) {
        printf("Usage: expect STEPS input integer parameter, defaulting to %d\n", STEPS);
    }
    else {
        STEPS = stoi(argv[1]);
        if (argc > 2) {
            nPoints = stoi(argv[2]);
        }
        if (argc > 3) {
            nFeatures = stoi(argv[3]);
        }
        if (argc > 4) {
            minPts = stoi(argv[4]);
        }
        if (argc > 5) {
            eps = stof(argv[5]);
        }
    }

    double* data;
    double lBound = 0.0;
    double rBound = 10.0;

    int repeat = 1;
    double t1 = 0, t2 = 0;

    FILE* fptr;
    fptr = fopen("perf_output.csv", "w");
    if (fptr == NULL) {
        printf("Error!");
        exit(1);
    }

    FILE* fptr1;
    fptr1 = fopen("runtimes.csv", "w");
    if (fptr1 == NULL) {
        printf("Error!");
        exit(1);
    }

    int i, j;
    double MOPS = 0.0;
    double time;
    for (i = 0; i < STEPS; i++) {

        std::vector<std::array<dtype, dataDim>> data_train = gen_data_x(nPoints_train);
        std::vector<size_t> labels = gen_data_y(nPoints_train);

        std::vector<std::array<dtype, dataDim>> data_test = gen_data_x(nPoints);

        std::cout << "TRAIN DATA SIZE: " << nPoints_train << std::endl;
        std::cout << "TEST DATA SIZE: " << nPoints << std::endl;

        /* Warm up cycle */
        std::vector<size_t> predictions = run_knn(data_train, labels, data_test);

        t1 = timer_rdtsc();
        for (j = 0; j < repeat; j++) {
            run_knn(data_train, labels, data_test);
        }
        t2 = timer_rdtsc();

        MOPS = (nPoints * repeat / 1e6) / ((double)(t2 - t1) / getHz());
        time = ((double)(t2 - t1) / getHz());

        printf("ERF: Native-C-VML: Size: %ld MOPS: %.6lf\n", nPoints, MOPS);
        std::cout << "TIME: " << time << std::endl;
        fflush(stdout);
        fprintf(fptr, "%ld,%.6lf\n", nPoints, MOPS);
        fprintf(fptr1, "%ld,%.6lf\n", nPoints, time);

        nPoints = nPoints * 2;
        if (repeat > 2)repeat -= 2;
    }
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
