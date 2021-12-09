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

#include <CL/sycl.hpp>

#include <dbscan.hpp>
#include <rdtsc.h>

using namespace std;
using namespace cl::sycl;

template <typename T>
void ReadInputFromBinFile (char const * filename, char* data_ptr, size_t data_size) {
    ifstream file;
    file.open(filename, ios::in|ios::binary);
    if (file) {
      file.read(data_ptr, data_size*sizeof(T));
      file.close();
    } else {
      std::cout << "Input file - " << filename << " not found.\n";
      exit(0);
    }
}

void WriteOutputToTextFile (char const * filename, double* data_ptr, size_t data_size) {
    ofstream file;
    file.open(filename, ios::out);
    if (file) {
      for (size_t i = 0; i < data_size; i++) {
	file << *data_ptr << std::endl;
      }
      file.close();
    } else {
      std::cout << "Input file - " << filename << " not found.\n";
      exit(0);
    }
}

double* readData(int nPoints, int nFeatures) {
    int arrSize = nPoints * nFeatures;
    double *data = new double[arrSize];

    ReadInputFromBinFile<double> ("X.bin", reinterpret_cast<char *>(data), arrSize);

    // ifstream dataFile(dataFileName);
    // if(!dataFile.is_open()) throw std::runtime_error("Could not open file");

    // string line;
    // double val;
    // int idx = 0;
    // while(getline(dataFile, line)) {
    //     std::stringstream ss(line);
    //     while(ss >> val) {
    //         data[idx] = val;
    //         if(ss.peek() == ',') ss.ignore();
    //         idx++;
    //     }
    // }
    // dataFile.close();
    // if(idx != arrSize) throw std::runtime_error("File data size does not match array size");

    return data;
}

int main(int argc, char * argv[]) {
  size_t nPoints = pow(2, 9);
  size_t nFeatures = 2;
  size_t minPts = 3;
  double eps = 1.0;
  int repeat = 1;

  bool test = false;

  /* Read nopt number of options parameter from command line */
  if (argc >= 2) {
    sscanf(argv[1], "%lu", &nPoints);
  }
  if (argc >= 3) {
    sscanf(argv[2], "%lu", &nFeatures);
  }
  if (argc >= 4) {
    sscanf(argv[3], "%lu", &minPts);
  }
  if (argc >= 5) {
    sscanf(argv[4], "%lf", &eps);
  }
  if (argc >= 6) {
    sscanf(argv[5], "%d", &repeat);
  }
  if (argc == 7) {
    char test_str[] = "-t";
    if (strcmp(test_str, argv[6]) == 0) {
      test = true;
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

  queue *q = nullptr;

  try {
    q = new queue{cpu_selector()};
  } catch (sycl::exception &re) {
    std::cerr << "No GPU device found\n";
    exit(1);
  }

  size_t *assignments = new size_t[nPoints];
  double *data = readData(nPoints, nFeatures);

  /* Warm up cycle */
  size_t nClusters = dbscan_reference_no_mem_save(q, nPoints, nFeatures, data, eps, minPts, assignments);

  t1 = timer_rdtsc();
  for(int j = 0; j < repeat; j++) {
    nClusters = dbscan_reference_no_mem_save(q, nPoints, nFeatures, data, eps, minPts, assignments);
  }
  t2 = timer_rdtsc();

  double time = ((double) (t2 - t1) / getHz());
  double MOPS = (nPoints * repeat / 1e6) / time;

  printf("ERF: Native-C-VML: Size: %lu Dim: %lu Eps: %.4lf minPts: %lu NCluster: %lu TIME: %.6lf\n", nPoints, nFeatures, eps, minPts, nClusters, time);
  fflush(stdout);
  fprintf(fptr, "%lu,%lu,%.4lf,%lu,%lu,%.6lf\n", nPoints, nFeatures, eps, minPts, nClusters, MOPS);
  fprintf(fptr1, "%lu,%.6lf\n", nPoints, time);

  if (test) {
    ofstream file;
    file.open("assignments.bin", ios::out|ios::binary);
    if (file) {
      file.write(reinterpret_cast<char *>(assignments), nPoints*sizeof(size_t));
      file.close();
    } else {
      std::cout << "Unable to open output file.\n";
    }
  }

  delete [] data;
  delete [] assignments;
  fclose(fptr);
  fclose(fptr1);

  return 0;
}
