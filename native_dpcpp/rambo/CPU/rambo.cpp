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

#include <cmath>
#include <vector>
#include <random>

#include "rambo.h"

using namespace std;
using namespace cl::sycl;

std::mt19937 rand_generator(SEED);

double genRand() {
  int a = rand_generator() >> 5;
  int b = rand_generator() >> 6;
  double value = (a * 67108864.0 + b) / 9007199254740992.0;
  return value;
  //return (double) rand() / RAND_MAX;
}

vector<double> vectMultiply(vector<double> a, vector<double> b, size_t nPoints) {
    size_t nOut = a.size() / nPoints / SIZE3;
    vector<double> result(nPoints * nOut);

    size_t idx2;
    double res;
    for(size_t i = 0; i < nPoints; i++) {
        for(size_t j = 0; j < nOut; j++) {
            idx2 = i * nOut + j;
            res = a[idx2 * SIZE3] * b[idx2 * SIZE3];
            for(size_t k = 1; k < SIZE3; k++) {
                res -= a[idx2 * SIZE3 + k] * b[idx2 * SIZE3 + k];
            }
            result[idx2] = res;
        }
    }

    return result;
}

vector<double> getMomentumSum(vector<double> inputParticles, size_t nPoints) {
    size_t size2 = inputParticles.size() / nPoints / SIZE3;
    vector<double> momentumSum(nPoints * SIZE3);

    double sum;
    size_t idx2;
    for(size_t i = 0; i < nPoints; i++) {
        for(size_t k = 0; k < SIZE3; k++) {
            sum = 0.0;
            for(size_t j = 0; j < size2; j++) {
                idx2 = i * size2 + j;
                sum += inputParticles[idx2 * SIZE3 + k];
            }
            momentumSum[i * SIZE3 + k] = sum;
        }
    }

    return momentumSum;
}

vector<double> getMass(vector<double> inputParticles, size_t nPoints) {
    vector<double> mass(nPoints);

    double mom2, val;
    for(size_t i = 0; i < nPoints; i++) {
        mom2 = 0.0;
        for(size_t k = 1; k < SIZE3; k++) {
            val = inputParticles[i * SIZE3 + k];
            mom2 += val * val;
        }
        val = inputParticles[i * SIZE3];
        mass[i] = sqrt(val * val - mom2);
    }

    return mass;
}

vector<double> getCombinedMass(vector<double> inputParticles, size_t nPoints) {
    vector<double> momentumSum = getMomentumSum(inputParticles, nPoints);
    return getMass(momentumSum, nPoints);
}

vector<double> getInputs(int ecms, size_t nPoints) {
    vector<double> pa = {ecms / 2.0, 0.0, 0.0, ecms / 2.0};
    vector<double> pb = {ecms / 2.0, 0.0, 0.0, -ecms / 2.0};

    size_t size2 = 2;
    vector<double> inputParticles(nPoints * size2 * SIZE3);

    size_t idx2;
    for(size_t i = 0; i < nPoints; i++) {
        idx2 = i * size2;
        for(size_t k = 0; k < SIZE3; k++) {
            inputParticles[idx2 * SIZE3 + k] = pa[k];
        }
        idx2 += 1;
        for(size_t k = 0; k < SIZE3; k++) {
            inputParticles[idx2 * SIZE3 + k] = pb[k];
        }
    }

    return inputParticles;
}

double* getOutputMom2(queue* q, size_t nPoints, size_t nOut) {
    size_t commonSize = nPoints * nOut;
    vector<double> C1(commonSize);
    vector<double> F1(commonSize);
    vector<double> Q1(commonSize);

    size_t outputSize = nPoints * nOut * SIZE3;

    for(size_t i = 0; i < nPoints; i++) {
        for(size_t j = 0; j < nOut; j++) {
            size_t idx2 = i * nOut + j;

            C1[idx2] = genRand();
            F1[idx2] = genRand();
            Q1[idx2] = genRand() * genRand();
        }
    }
    
    double *d_C1Pointer = (double*) malloc_shared(commonSize * sizeof(double), *q);
    double *d_F1Pointer = (double*) malloc_shared(commonSize * sizeof(double), *q);
    double *d_Q1Pointer = (double*) malloc_shared(commonSize * sizeof(double), *q);
    double *d_outputPointer = (double*) malloc_shared(outputSize * sizeof(double), *q);
    
    q->memcpy(d_C1Pointer, C1.data(), commonSize * sizeof(double));
    q->memcpy(d_F1Pointer, F1.data(), commonSize * sizeof(double));
    q->memcpy(d_Q1Pointer, Q1.data(), commonSize * sizeof(double));

    q->wait();

    /* omp slows down the algorithm */
    // #pragma omp target teams distribute parallel for simd private(idx2, C, S, F, Q) map(to: C1Pointer[0:commonSize], F1Pointer[0:commonSize], Q1Pointer[0:commonSize]) map(from: outputPointer[0:outputSize])
    //for(size_t i = 0; i < nPoints; i++) {
    q->submit([&](handler& h) {
	h.parallel_for<class theKernel>(range<1>{nPoints}, [=](id<1> myID) {
	    for(size_t j = 0; j < nOut; j++) {
	      int i = myID[0];
	      size_t idx2 = i * nOut + j;

	      double C = 2.0 * d_C1Pointer[idx2] - 1.0;
	      double S = sqrt(1 - C * C);
	      double F = 2.0 * M_PI * d_F1Pointer[idx2];
	      double Q = -log(d_Q1Pointer[idx2]);

	      d_outputPointer[idx2 * SIZE3] = Q;
	      d_outputPointer[idx2 * SIZE3 + 1] = Q * S * sin(F);
	      d_outputPointer[idx2 * SIZE3 + 2] = Q * S * cos(F);
	      d_outputPointer[idx2 * SIZE3 + 3] = Q * C;
	    }
	  });
      });

    q->wait();

    double* output = new double[outputSize];
    q->memcpy(output, d_outputPointer, outputSize * sizeof(double));

    q->wait();

    return output;
}

double* generatePoints(queue* q, size_t ecms, size_t nPoints, size_t nOut) {
  double* outputParticles = getOutputMom2(q, nPoints, nOut);

  return outputParticles;

    // vector<double> inputParticles = getInputs(ecms, nPoints);
    // vector<double> inputMass = getCombinedMass(inputParticles, nPoints);

    // vector<double> outputMomSum = getMomentumSum(outputParticles, nPoints);
    // vector<double> outputMass = getMass(outputMomSum, nPoints);

    // double G, X, B, BQ, A, E, D, C1, C;

    // size_t inputParticlesSize2 = inputParticles.size() / nPoints / SIZE3;
    // size_t outputParticlesSize2 = outputParticles.size() / nPoints / SIZE3;
    // size_t pointsSize2 = inputParticlesSize2 + outputParticlesSize2;
    // vector<double> points(nPoints * pointsSize2 * SIZE3);

    // double val;
    // size_t idx2, idx3, pointsIdx2, pointsIdx3;
    // double res;
    // for(size_t i = 0; i < nPoints; i++) {
    //     for(size_t j = 0; j < inputParticlesSize2; j++) {
    //         idx2 = i * inputParticlesSize2 + j;
    //         pointsIdx2 = i * pointsSize2 + j;
    //         for(size_t k = 0; k < SIZE3; k++) {
    //             idx3 = idx2 * SIZE3 + k;
    //             pointsIdx3 = pointsIdx2 * SIZE3 + k;
    //             points[pointsIdx3] = inputParticles[idx3];
    //         }
    //     }
    //     for(size_t j = 0; j < nOut; j++) {
    //         idx2 = i * nOut + j;

    //         G = outputMomSum[i * SIZE3] / outputMass[i];
    //         X = inputMass[i] / outputMass[i];

    //         res = 0.0;
    //         for(size_t k = 1; k < SIZE3; k++) {
    //             idx3 = idx2 * SIZE3 + k;

    //             B = -outputMomSum[i * SIZE3 + k] / outputMass[i];
    //             res -= B * outputParticles[idx3];
    //         }
    //         BQ = -1.0 * res;

    //         A = 1.0 / (1.0 + G);
    //         E = outputParticles[idx2 * SIZE3];
    //         D = G * E + BQ;
    //         C1 = E + A * BQ;

    //         pointsIdx2 = i * pointsSize2 + j + inputParticlesSize2;
    //         pointsIdx3 = pointsIdx2 * SIZE3;
    //         points[pointsIdx3] = X * D;
    //         for(size_t k = 1; k < SIZE3; k++) {
    //             idx3 = idx2 * SIZE3 + k;
    //             pointsIdx3 = pointsIdx2 * SIZE3 + k;

    //             B = -outputMomSum[i * SIZE3 + k] / outputMass[i];
    //             C = outputParticles[idx3] + B * C1;
    //             points[pointsIdx3] = X * C;
    //         }
    //     }
    // }

    // return points;
}

double* rambo(queue* q, size_t nPoints) {
  rand_generator.seed(SEED);
  size_t ecms = 100;
  size_t nOut = NOUT;
  
  return generatePoints(q, ecms, nPoints, nOut);
    
    // size_t eSize2 = e.size() / nPoints / SIZE3;
    // vector<double> h(SIZE3 * nPoints);

    // size_t idx2, idx3;
    // double max, val;
    // for(size_t i = 0; i < SIZE3; i++) {
    //     for(size_t j = 0; j < nPoints; j++) {
    //         idx2 = j * eSize2 + 2;
    //         idx3 = idx2 * SIZE3 + i;
    //         max = e[idx3];
    //         for(size_t k = 3; k < 5; k++) {
    //             idx2 = j * eSize2 + k;
    //             idx3 = idx2 * SIZE3 + i;
    //             val = e[idx3];
    //             if(val > max) {
    //                 max = val;
    //             }
    //         }
    //         h[i * nPoints + j] = max;
    //     }
    // }
}
