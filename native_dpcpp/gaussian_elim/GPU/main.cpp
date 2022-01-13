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
#include "gaussianElim.h"
#include "rdtsc.h"
#include <sstream>
#include <fstream>

#define BLOCK_SIZE_0 256
#define BLOCK_SIZE_1_X 16
#define BLOCK_SIZE_1_Y 16


auto InitData(size_t data_size, std::string filename)
{
    auto data = std::make_unique<tfloat[]>(data_size);

    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file)
    {
        file.read(reinterpret_cast<char *>(data.get()), data_size * sizeof(tfloat));
        file.close();
    }
    else
    {
        std::cout << "Input file not found.\n";
        exit(0);
    }

    return data;
}


void BackSub(tfloat *a, tfloat *b, tfloat *finalVec, int size)
{
  // solve "bottom up"
  int i, j;
  for(i = 0; i < size; i++)
  {
    finalVec[size-i-1] = b[size-i-1];
    for(j = 0; j < i; j++)
    {
      finalVec[size-i-1] -= *(a + size * (size - i - 1) + (size - j - 1)) * finalVec[size - j - 1];
    }
    finalVec[size - i - 1] = finalVec[size - i - 1] / *(a + size * (size - i - 1) + (size - i - 1));
  }
}


int main(int argc, char *argv[])
{
    int repeat = 1;
    size_t matrix_size = pow(2, 2);

    bool test = false;
    double time;

    /* Read number of options parameter from command line */
    if (argc >= 2)
    {
        sscanf(argv[1], "%lu", &matrix_size);
    }
    if (argc >= 3)
    {
        sscanf(argv[2], "%d", &repeat);
    }
    if (argc == 4)
    {
      char test_str[] = "-t";
      if (strcmp(test_str, argv[3]) == 0)
      {
	    test = true;
      }
    }

    FILE *fptr;
    fptr = fopen("perf_output.csv", "a");
    if (fptr == NULL)
    {
        printf("Error!");
        exit(1);
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "a");
    if (fptr1 == NULL)
    {
        printf("Error!");
        exit(1);
    }

    queue *q = nullptr;
    try
    {
        q = new queue{gpu_selector()};
    }
    catch (sycl::exception &re)
    {
        std::cerr << "No GPU device found\n";
        exit(1);
    }

    auto matrix_ptr = InitData(matrix_size*matrix_size, "m_data.bin");
    tfloat *m = matrix_ptr.get();

    auto b_ptr = InitData(matrix_size, "v_data.bin");
    tfloat *b = b_ptr.get();

    auto extra_matrix_ptr = std::make_unique<tfloat[]>(matrix_size*matrix_size);
    tfloat *extra_matrix = extra_matrix_ptr.get();

    auto final_vec_ptr = std::make_unique<tfloat[]>(matrix_size);
    tfloat *final_vec = final_vec_ptr.get();

    ///////////////////////
    /* Determine block size */
    int globalWorksizeFan1[1];
    int globalWorksizeFan2[2];
    int localWorksizeFan1Buf[1] = {BLOCK_SIZE_0};
    int localWorksizeFan2Buf[2] = {BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y};
    int *localWorksizeFan1 = NULL;
    int *localWorksizeFan2 = NULL;

    globalWorksizeFan1[0] = matrix_size;
    globalWorksizeFan2[0] = matrix_size;
    globalWorksizeFan2[1] = matrix_size;

    if(localWorksizeFan1Buf[0])
    {
      localWorksizeFan1 = localWorksizeFan1Buf;
      globalWorksizeFan1[0] = (int)ceil(globalWorksizeFan1[0] / (double)localWorksizeFan1Buf[0]) * localWorksizeFan1Buf[0];
    }
    if(localWorksizeFan2Buf[0])
    {
      localWorksizeFan2 = localWorksizeFan2Buf;
      globalWorksizeFan2[0] = (int)ceil(globalWorksizeFan2[0] / (double)localWorksizeFan2Buf[0]) * localWorksizeFan2Buf[0];
      globalWorksizeFan2[1] = (int)ceil(globalWorksizeFan2[1] / (double)localWorksizeFan2Buf[1]) * localWorksizeFan2Buf[1];
    }
    ///////////////////////

    for (int i = 0; i < matrix_size* matrix_size; i++)
    {
      std::cout << m[i] << " ";
    }

    std::cout << " " << std::endl;

    /* Warm up cycle */
    ForwardSub(q, m, b, extra_matrix, matrix_size, globalWorksizeFan1, localWorksizeFan1Buf, globalWorksizeFan2, localWorksizeFan2Buf);

    for (int i = 0; i < matrix_size* matrix_size; i++)
    {
      std::cout << m[i] << " ";
    }

    std::cout << " " << std::endl;

    /* Compute the result */
    BackSub(m, b, final_vec, matrix_size);

    for (int i = 0; i < matrix_size; i++)
    {
      std::cout << final_vec[i] << " ";
    }

    std::cout << " " << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < repeat; j++)
    {
      ForwardSub(q, m, b, extra_matrix, matrix_size, globalWorksizeFan1, localWorksizeFan1Buf, globalWorksizeFan2, localWorksizeFan2Buf);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = t2 - t1;

    time = elapsed_seconds.count() / repeat;

    double MOPS = (matrix_size * repeat / 1e6) / ((double)time / getHz());

    printf("ERF: Native-C-VML: Size: %ld Time: %.6lf\n", matrix_size, time);
    fflush(stdout);
    fprintf(fptr, "%ld,%.6lf\n", matrix_size, MOPS);
    fprintf(fptr1, "%ld,%.6lf\n", matrix_size, time);

    fclose(fptr);
    fclose(fptr1);

    return 0;
}
