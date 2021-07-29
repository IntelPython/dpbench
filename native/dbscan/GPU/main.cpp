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

#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdexcept>
#include <sstream>

#include <dbscan.hpp>
#include <rdtsc.h>

using namespace std;

const string dataFileName = "data.csv";

static int stoi(char *h) {
    stringstream in (h);
    int res;
    in >> res;
    return res;
}

static double stof(char *h) {
    stringstream in (h);
    double res;
    in >> res;
    return res;
}

double* readData(int nPoints, int nFeatures) {
    int arrSize = nPoints * nFeatures;
    double *data = new double[arrSize];

    ifstream dataFile(dataFileName);
    if(!dataFile.is_open()) throw std::runtime_error("Could not open file");

    string line;
    double val;
    int idx = 0;
    while(getline(dataFile, line)) {
        std::stringstream ss(line);
        while(ss >> val) {
            data[idx] = val;
            if(ss.peek() == ',') ss.ignore();
            idx++;
        }
    }
    dataFile.close();
    if(idx != arrSize) throw std::runtime_error("File data size does not match array size");

    return data;
}

int main(int argc, char * argv[]) {
    int STEPS = 6;
    int nPoints = pow(2, 9);
    int nFeatures = 2;
    int minPts = 3;
    double eps = 1.0;
    int repeat = 1;

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
            if (2 * nFeatures > minPts) {
                minPts = 2 * nFeatures;
            }
        }
        if (argc > 4) {
            minPts = stoi(argv[4]);
        }
        if (argc > 5) {
            eps = stof(argv[5]);
        }
        if (argc > 6) {
            repeat = stof(argv[6]);
        }
    }

    double lBound = 0.0;
    double rBound = 10.0;

    clock_t t1 = 0, t2 = 0;

    FILE *fptr;
    fptr = fopen("perf_output.csv", "a");
    if(fptr == NULL) {
      printf("Error!");
      exit(1);
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "a");
    if(fptr1 == NULL) {
      printf("Error!");
      exit(1);
    }

    int nClusters;
    double time, MOPS;
    int *assignments = new int[nPoints];
    double *data = readData(nPoints, nFeatures);

    /* Warm up cycle */
    nClusters = dbscan_reference_no_mem_save(nPoints, nFeatures, data, eps, minPts, assignments);

    t1 = timer_rdtsc();
    for(int j = 0; j < repeat; j++) {
        nClusters = dbscan_reference_no_mem_save(nPoints, nFeatures, data, eps, minPts, assignments);
    }
    t2 = timer_rdtsc();

    time = ((double) (t2 - t1) / getHz());
    MOPS = (nPoints * repeat / 1e6) / time;

    printf("ERF: Native-C-VML: Size: %d Dim: %d Eps: %.4lf minPts: %d NCluster: %d TIME: %.6lf\n", nPoints, nFeatures, eps, minPts, nClusters, time);
    fflush(stdout);
    fprintf(fptr, "%d,%d,%.4lf,%d,%d,%.6lf\n", nPoints, nFeatures, eps, minPts, nClusters, MOPS);
    fprintf(fptr1, "%d,%d,%.4lf,%d,%d,%.6lf\n", nPoints, nFeatures, eps, minPts, nClusters, time);

    delete [] data;
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
