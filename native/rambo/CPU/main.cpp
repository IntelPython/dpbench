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

#include <sstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <rambo_wo_mkl.cpp>

static int stoi (char *h) {
    stringstream in (h);
    int res;
    in >> res;
    return res;
}

static double stof (char *h) {
    stringstream in (h);
    double res;
    in >> res;
    return res;
}

int main(int argc, char * argv[]) {
    int repeat = 1;
    int STEPS = 7;
    size_t nPoints = pow(2, 13);

    if (argc < 2) {
        cout << "Usage: expect STEPS input integer parameter, defaulting to " << STEPS << endl;
    }
    else {
        STEPS = stoi(argv[1]);
        if (argc > 2) {
            nPoints = stoi(argv[2]);
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

    double time, MOPS;
    for(int i = 0; i < STEPS; i++) {
        /* Warm up cycle */
        rambo(nPoints);

        auto start = chrono::high_resolution_clock::now();
        for(int j = 0; j < repeat; j++) {
            rambo(nPoints);
        }
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> diff = end - start;
        time = diff.count();
        MOPS = nPoints * repeat / 1e6 / time;

        cout << "ERF: Native-C-VML: Size: " << nPoints << " time: " << time << endl;
        fflush(stdout);
        fprintf(fptr, "%zu,%.6lf\n", nPoints, MOPS);
        fprintf(fptr1, "%zu,%.6lf\n", nPoints, time);

        nPoints = nPoints * 2;
	if (repeat > 2) repeat -= 2;
    }
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
