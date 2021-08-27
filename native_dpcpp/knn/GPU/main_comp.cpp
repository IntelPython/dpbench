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
#include <fstream>

#define SEED 7777777

int stoi(char *h)
{
    std::stringstream in(h);
    int res;
    in >> res;
    return res;
}

double stof(char *h)
{
    std::stringstream in(h);
    double res;
    in >> res;
    return res;
}

double rand32(double a, double b)
{
    return abs((rand() << 16) | rand()) % 1000000000 / 1000000000.0 * (b - a) + a;
}

double *gen_data_x(size_t data_size)
{
    double *data = new double[data_size * DATADIM];

    for (size_t i = 0; i < data_size; ++i)
    {
        for (size_t j = 0; j < DATADIM; ++j)
        {
            data[i * DATADIM + j] = rand32(0, 1);
        }
    }

    return data;
}

auto read_data_x(size_t data_size, std::string filename)
{

    auto n = data_size * DATADIM;
    auto data = std::make_unique<double_t[]>(n);

    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file)
    {
        file.read(reinterpret_cast<char *>(data.get()), n * sizeof(double));
        file.close();
    }
    else
    {
        std::cout << "Input file not found.\n";
        exit(0);
    }

    return data;
}

std::vector<size_t> gen_data_y(size_t data_size)
{
    std::vector<size_t> labels;
    for (size_t i = 0; i < data_size; ++i)
    {
        labels.push_back((size_t)rand() % NUM_CLASSES);
    }

    return labels;
}

auto read_data_y(size_t data_size, std::string filename)
{

    auto n = data_size;
    auto data = std::make_unique<size_t[]>(n);

    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file)
    {
        file.read(reinterpret_cast<char *>(data.get()), n * sizeof(size_t));
        file.close();
    }
    else
    {
        std::cout << "Input file not found.\n";
        exit(0);
    }

    return data;
}

int main(int argc, char *argv[])
{
    int repeat = 1;
    double t1 = 0, t2 = 0;

    size_t nPoints_train = pow(2, 10);
    size_t nPoints = pow(2, 10);

    size_t nFeatures = 1;
    size_t minPts = 5;
    double eps = 1.0;

    /* Read number of options parameter from command line */
    if (argc >= 2)
    {
        sscanf(argv[1], "%lu", &nPoints);
    }
    if (argc >= 3)
    {
        sscanf(argv[2], "%d", &repeat);
    }

    FILE *fptr;
    fptr = fopen("perf_output.csv", "w");
    if (fptr == NULL)
    {
        printf("Error!");
        exit(1);
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "w");
    if (fptr1 == NULL)
    {
        printf("Error!");
        exit(1);
    }

    srand(SEED);

    queue *q = nullptr;
    try
    {
        q = new queue{gpu_selector()};
    }
    catch (runtime_error &re)
    {
        std::cerr << "No GPU device found\n";
        exit(1);
    }

    int i, j;
    double MOPS = 0.0;
    double time;

    auto data_train_ptr = read_data_x(nPoints_train, "x_train.bin");
    double *data_train = data_train_ptr.get();

    auto train_labels_ptr = read_data_y(nPoints_train, "y_train.bin");
    size_t *train_labels = train_labels_ptr.get();

    auto data_test_ptr = read_data_x(nPoints, "x_test.bin");
    double *data_test = data_test_ptr.get();

    size_t *predictions = new size_t[nPoints];
    double *votes_to_classes = (double *)calloc(nPoints * NUM_CLASSES, sizeof(double));


    double *d_votes_to_classes = (double *)malloc_device(test_size * NUM_CLASSES * sizeof(double), *q);
    double *queue_neighbors = (double *)malloc_device(test_size * NEAREST_NEIGHS * 2 * sizeof(double), *q);

    double *d_train = (double *)malloc_device(train_nrows * DATADIM * sizeof(double), *q);
    size_t *d_train_labels = (size_t *)malloc_device(train_nrows * sizeof(size_t), *q);
    double *d_test = (double *)malloc_device(test_size * DATADIM * sizeof(double *), *q);
    size_t *d_predictions = (size_t *)malloc_device(test_size * sizeof(size_t), *q);

    // copy data host to device
    q->memcpy(d_train, train, train_nrows * DATADIM * sizeof(double));
    q->memcpy(d_train_labels, train_labels, train_nrows * sizeof(size_t));
    q->memcpy(d_test, test, test_size * DATADIM * sizeof(double));
    q->memcpy(d_votes_to_classes, votes_to_classes, test_size * NUM_CLASSES * sizeof(double));
    q->wait();

    /* Warm up cycle */
    run_knn(q, data_train, train_labels, data_test, nPoints_train, nPoints, predictions, votes_to_classes);

    t1 = timer_rdtsc();
    for (j = 0; j < repeat; j++)
    {
        run_knn(q, data_train, train_labels, data_test, nPoints_train, nPoints, predictions, votes_to_classes);
    }
    t2 = timer_rdtsc();

    MOPS = (nPoints * repeat / 1e6) / ((double)(t2 - t1) / getHz());
    time = ((double)(t2 - t1) / getHz());

    printf("ERF: Native-C-VML: Size: %ld Time: %.6lf\n", nPoints, time);
    fflush(stdout);
    fprintf(fptr, "%ld,%.6lf\n", nPoints, MOPS);
    fprintf(fptr1, "%ld,%.6lf\n", nPoints, time);

    fclose(fptr);
    fclose(fptr1);

    q->memcpy(predictions, d_predictions, test_size * sizeof(size_t));

    delete[] predictions;

    free(d_train, q->get_context());
    free(d_test, q->get_context());
    free(d_train_labels, q->get_context());
    free(d_predictions, q->get_context());
    free(d_votes_to_classes, q->get_context());

    return 0;
}
