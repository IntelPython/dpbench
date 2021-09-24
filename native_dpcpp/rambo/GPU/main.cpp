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
#include <chrono>

#include "rambo.h"

using namespace std;
using namespace cl::sycl;

int main(int argc, char * argv[]) {
    int repeat = 1;
    int STEPS = 7;
    size_t nPoints = pow(2, 20);

    bool test = false;

    if (argc >= 2) {
      sscanf(argv[1], "%lu", &nPoints);
    }
    if (argc >= 3) {
      sscanf(argv[2], "%d", &repeat);
    }
    if (argc == 4) {
      char test_str[] = "-t";
      if (strcmp(test_str, argv[3]) == 0) {
	test = true;
      }
    }

    FILE *fptr;
    fptr = fopen("perf_output.csv", "w");
    if(fptr == NULL) {
      printf("Error!");
      exit(1);
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "w");
    if(fptr1 == NULL) {
      printf("Error!");
      exit(1);
    }

    queue *q = nullptr;

    try {
      q = new queue{gpu_selector()};
    } catch (sycl::exception &re) {
      std::cerr << "No GPU device found\n";
      exit(1);
    }

    /* Warm up cycle */
    double * output = rambo(q, nPoints);

    auto start = chrono::high_resolution_clock::now();
    for(int j = 0; j < repeat; j++) {
      output = rambo(q, nPoints);
    }
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> diff = end - start;
    double time = diff.count();
    double MOPS = nPoints * repeat / 1e6 / time;

    std::cout << "ERF: Native-C-VML: Size: " << nPoints << " time: " << time << std::endl;
    fflush(stdout);
    fprintf(fptr, "%zu,%.6lf\n", nPoints, MOPS);
    fprintf(fptr1, "%zu,%.6lf\n", nPoints, time);

    fclose(fptr);
    fclose(fptr1);

    if (test) {
      ofstream file;
      file.open("output.bin", ios::out|ios::binary);
      if (file) {
	file.write(reinterpret_cast<char *>(output), (nPoints * NOUT * SIZE3)*sizeof(double));
	file.close();
      } else {
	std::cout << "Unable to open output file.\n";
      }
    }

    return 0;
}
