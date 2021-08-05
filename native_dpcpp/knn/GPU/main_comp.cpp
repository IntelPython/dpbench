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
#include "knn.h"
#include "rdtsc.h"
#include <sstream>

#define SEED 7777777

int stoi(char* h) {
    std::stringstream in(h);
    int res;
    in >> res;
    return res;
}

double stof(char* h) {
    std::stringstream in(h);
    double res;
    in >> res;
    return res;
}

double rand32(double a, double b) {
    return abs((rand() << 16) | rand()) % 1000000000 / 1000000000.0 * (b - a) + a;
}


double* gen_data_x(size_t data_size, queue* q)
{
  double* data = (double*)malloc_shared(data_size * DATADIM * sizeof(double), *q);

  for (size_t i = 0; i < data_size; ++i) {
    for (size_t j = 0; j < DATADIM; ++j){
      data[i*DATADIM + j] = rand32(0, 1);
    }
  }

  return data;
}

size_t* gen_data_y(size_t data_size, queue* q)
{
  size_t* labels = (size_t*)malloc_shared(data_size * sizeof(size_t), *q);
  for (size_t i = 0; i < data_size; ++i) {
    labels[i] = (size_t)(rand() % NUM_CLASSES);
  }

  return labels;
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

    srand(SEED);

    queue *q = nullptr;
    try {
      q = new queue{gpu_selector()};
    } catch (runtime_error &re) {
      std::cerr << "No GPU device found\n";
      exit(1);
    }    

    int i, j;
    double MOPS = 0.0;
    double time;
    for (i = 0; i < STEPS; i++) {

      double* data_train = gen_data_x(nPoints_train, q);
      size_t* train_labels = gen_data_y(nPoints_train, q);
      double* data_test = gen_data_x(nPoints, q);
      size_t* predictions = (size_t*)malloc_shared(nPoints * sizeof(size_t), *q);

      /* Warm up cycle */
      run_knn(q, data_train, train_labels, data_test, nPoints_train, nPoints, predictions);

      t1 = timer_rdtsc();
      for (j = 0; j < repeat; j++) {
	run_knn(q, data_train, train_labels, data_test, nPoints_train, nPoints, predictions);
      }
      t2 = timer_rdtsc();

      MOPS = (nPoints * repeat / 1e6) / ((double)(t2 - t1) / getHz());
      time = ((double)(t2 - t1) / getHz());

#if 0
      for (size_t j = 0; j < nPoints; j++) {
	printf("%lu\n", predictions[i]);
      }
#endif

      printf("ERF: Native-C-VML: Size: %ld Time: %.6lf\n", nPoints, time);
      fflush(stdout);
      fprintf(fptr, "%ld,%.6lf\n", nPoints, MOPS);
      fprintf(fptr1, "%ld,%.6lf\n", nPoints, time);

      free(data_train, q->get_context());
      free(train_labels, q->get_context());
      free(data_test, q->get_context());
      free(predictions, q->get_context());

      nPoints = nPoints * 2;
      if (repeat > 2) repeat -= 2;
    }
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
