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
#include <rdtsc.h>
#include <sstream>

#include <stdexcept>
#include <string>
#include <fstream>
#include <chrono>

#include <knn.hpp>


// std=c++11 or higher (c++14, c++17)
using dtype = double;
const int nFeatures = 16;


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


dtype* read_data_x(int nPoints, int nFeatures, std::string dataFileName) {
    int arrSize = nPoints * nFeatures;
    dtype* data = new dtype[arrSize];

    std::ifstream dataFile(dataFileName);
    if (!dataFile.is_open()) throw std::runtime_error("Could not open file");

    std::string line;
    double val;
    int idx = 0;
    while (getline(dataFile, line)) {
        std::stringstream ss(line);
        while (ss >> val) {
            data[idx] = val;
            if (ss.peek() == ',') ss.ignore();
            idx++;
        }
    }
    dataFile.close();
    if (idx != arrSize) throw std::runtime_error("File data size does not match array size");

    return data;
}

int* read_data_y(int nPoints, std::string dataFileName) {
    int arrSize = nPoints;
    int* data = new int[arrSize];

    std::ifstream dataFile(dataFileName);
    if (!dataFile.is_open()) throw std::runtime_error("Could not open file");

    std::string line;
    int val;
    int idx = 0;
    while (getline(dataFile, line)) {
        std::stringstream ss(line);
        while (ss >> val) {
            data[idx] = val;
            idx++;
        }
    }
    dataFile.close();
    if (idx != arrSize) throw std::runtime_error("File data size does not match array size");

    return data;
}

dtype* gen_data_x(int nPoints, int nFeatures)
{
    int arrSize = nPoints * nFeatures;
    dtype* data = new dtype[arrSize];

    for (int i = 0; i < arrSize; ++i)
    {
        data[i] = rand32(0, 1);
    }

    return data;
}

int* gen_data_y(int nPoints)
{
    int* labels = new int[nPoints];
    for (int i = 0; i < nPoints; ++i)
    {
        labels[i] = (int)rand() % classesNum;
    }

    return labels;
}

void write_predictions(int* predictions, int size, std::string fileName)
{
    std::ofstream outFile(fileName);

    for (int i = 0; i < size; ++i)
    {
        outFile << predictions[i] << "\n";
    }
}


int main(int argc, char* argv[]) {
    int STEPS = 1;
    srand(0);

    int nPoints_train = pow(2, 9);
    int nPoints = pow(2, 15);

    std::cout << "TRAIN SIZE: " << nPoints_train << std::endl;
    std::cout << "TEST SIZE: " << nPoints << std::endl;

    int repeat = 100;

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
    double MOPS;
    double time;
    for (i = 0; i < STEPS; i++) {

        dtype* data_train = gen_data_x(nPoints_train, nFeatures);
        int* labels = gen_data_y(nPoints_train);
        dtype* data_test = gen_data_x(nPoints, nFeatures);

        std::cout << "TRAIN_SIZE: " << nPoints_train << std::endl;
        std::cout << "TEST_SIZE: " << nPoints << std::endl;

        // Define CL device, allocate buffers
        KNN knnAlg;
        knnAlg.init();
        knnAlg.allocate_buffers(nPoints_train, nPoints);

        // Map buffers ToDo: remove copying
        auto train_ptr = knnAlg.map_train(nPoints_train);
        for (int k = 0; k < nPoints_train * nFeatures; ++k)
        {
            train_ptr[k] = data_train[k];
        }
        auto test_ptr = knnAlg.map_test(nPoints);
        for (int k = 0; k < nPoints * nFeatures; ++k)
        {
            test_ptr[k] = data_test[k];
        }
        auto labels_ptr = knnAlg.map_labels(nPoints_train);
        for (int k = 0; k < nPoints_train; ++k)
        {
           labels_ptr[k] = labels[k];
        }
        auto predictions_ptr = knnAlg.map_predictions(nPoints);

        /* Warm up cycle */
        knnAlg.run_knn_opencl(train_ptr, test_ptr, labels_ptr, predictions_ptr, nPoints_train, nPoints);

        auto start = std::chrono::high_resolution_clock::now();
        for (j = 0; j < repeat; j++) {
            knnAlg.run_knn_opencl(train_ptr, test_ptr, labels_ptr, predictions_ptr, nPoints_train, nPoints);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;

        //write_predictions(predictions_ptr, nPoints, "y_test.csv");
        time = elapsed_seconds.count() / repeat;

        MOPS = (nPoints * repeat / 1e6) / (time);

        printf("ERF: Native-C-VML: Size: %d MOPS: %.6lf\n", nPoints, MOPS);
        std::cout << "TIME: " << time << std::endl;

        fflush(stdout);
        fprintf(fptr, "%d,%.6lf\n", nPoints, MOPS);
        fprintf(fptr1, "%d,%.6lf\n", nPoints, time);

        nPoints = nPoints * 2;
        repeat -= 2;

        delete[] data_train;
        delete[] labels;
        delete[] data_test;

    }
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
