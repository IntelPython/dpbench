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

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <array>
#include <memory>
#include "mkl.h"
#include "omp.h"

using namespace std;

constexpr int INPUTSIZE2 = 2;
constexpr int SIZE3 = 4;

unique_ptr<double[]> getMomentumSum(unique_ptr<double[]>& particles, size_t nPoints, int nOut) {
    unique_ptr<double[]> momentumSum(new double[nPoints * SIZE3]);

    #pragma omp parallel for simd
    for(size_t i = 0; i < nPoints; i++) {
        double sum;
        size_t idx2;
        for(int k = 0; k < SIZE3; k++) {
            sum = 0.0;
            for(int j = 0; j < nOut; j++) {
                idx2 = i * nOut + j;
                sum += particles[idx2 * SIZE3 + k];
            }
            momentumSum[i * SIZE3 + k] = sum;
        }
    }

    return momentumSum;
}

unique_ptr<double[]> getMass(unique_ptr<double[]>& momentumSum, size_t nPoints) {
    unique_ptr<double[]> mass(new double[nPoints]);

    #pragma omp parallel for simd
    for(size_t i = 0; i < nPoints; i++) {
        double val;
        double mom2 = 0.0;
        for(int k = 1; k < SIZE3; k++) {
            val = momentumSum[i * SIZE3 + k];
            mom2 += val * val;
        }
        val = momentumSum[i * SIZE3];
        mass[i] = sqrt(val * val - mom2);
    }

    return mass;
}

unique_ptr<double[]> getCombinedMass(unique_ptr<double[]>& particles, size_t nPoints) {
    auto momentumSum = getMomentumSum(particles, nPoints, INPUTSIZE2);
    return getMass(momentumSum, nPoints);
}

unique_ptr<double[]> getInputs(int ecms, size_t nPoints) {
    array<double, SIZE3> pa = {ecms / 2.0, 0.0, 0.0, ecms / 2.0};
    array<double, SIZE3> pb = {ecms / 2.0, 0.0, 0.0, -ecms / 2.0};

    unique_ptr<double[]> inputParticles(new double[nPoints * INPUTSIZE2 * SIZE3]);

    #pragma omp parallel for simd
    for(size_t i = 0; i < nPoints; i++) {
        size_t idx2 = i * INPUTSIZE2;
        for(int k = 0; k < SIZE3; k++) {
            inputParticles[idx2 * SIZE3 + k] = pa[k];
        }
        idx2 += 1;
        for(int k = 0; k < SIZE3; k++) {
            inputParticles[idx2 * SIZE3 + k] = pb[k];
        }
    }

    return inputParticles;
}

unique_ptr<double[]> getOutputMom2(const size_t nPoints, const size_t nOut) {
    const size_t size12 = nPoints * nOut;
    constexpr double a = 0.0, b = 1.0;
    constexpr int buffSize = 1 << 10;

    unique_ptr<double[]> output(new double[size12 * SIZE3]);
    mkl_domain_set_num_threads(1, MKL_DOMAIN_VML);

    const int numStreams = omp_get_max_threads();
    VSLStreamStatePtr streams[numStreams];
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        vslNewStream(&streams[tid], VSL_BRNG_MT2203 + tid, 777);
        #pragma omp for
        for(size_t j = 0; j < (size12 + buffSize - 1) / buffSize; ++j) {
            const int tid = omp_get_thread_num();
            const size_t low = j * buffSize;
            const size_t high = min<size_t>((j + 1) * buffSize, size12);
            const size_t blockSize = high - low;

            array<double, buffSize> localC1 = {};
            array<double, buffSize> localF1 = {};
            array<double, buffSize> localF2 = {};
            array<double, buffSize> localQ0 = {};
            array<double, buffSize> localQ1 = {};

            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, streams[tid], blockSize, localC1.data(), 2.0 * a - 1.0, 2.0 * b - 1.0);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, streams[tid], blockSize, localF1.data(), a * 2.0 * M_PI, b * 2.0 * M_PI);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, streams[tid], blockSize, localQ0.data(), a, b);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, streams[tid], blockSize, localQ1.data(), a, b);

            vdMul(blockSize, localQ0.data(), localQ1.data(), localQ1.data());
            vdLn(blockSize, localQ1.data(), localQ1.data());

            vdMul(blockSize, localC1.data(), localC1.data(), localQ0.data());
            vdLinearFrac(blockSize, localQ0.data(), localQ0.data(), -1.0 , 1.0, 0.0, 1.0, localQ0.data());
            vdSqrt(blockSize, localQ0.data(), localQ0.data());

            vdSinCos(blockSize, localF1.data(), localF1.data(), localF2.data());

            for(int k = 0; k < blockSize; ++k) {
                const double C = localC1[k];
                const double S = localQ0[k];
                const double sinF = localF1[k];
                const double cosF = localF2[k];
                const double Q = -localQ1[k];
                const double QS = Q * S;

                const int idx = (low + k) * SIZE3;
                output[idx] = Q;
                output[idx + 1] = QS * sinF;
                output[idx + 2] = QS * cosF;
                output[idx + 3] = Q * C;
            }
        }
        vslDeleteStream(&streams[tid]);
    }

    return output;
}

unique_ptr<double[]> generatePoints(size_t ecms, size_t nPoints, size_t nOut) {
    auto inputParticles = getInputs(ecms, nPoints);
    auto inputMass = getCombinedMass(inputParticles, nPoints);
    auto outputParticles = getOutputMom2(nPoints, nOut);
    auto outputMomSum = getMomentumSum(outputParticles, nPoints, nOut);
    auto outputMass = getMass(outputMomSum, nPoints);

    unique_ptr<double[]> points(new double[nPoints * nOut * SIZE3]);

    #pragma omp parallel for
    for(size_t i = 0; i < nPoints; i++) {
        size_t idx2, idx3;
        const double G = outputMomSum[i * SIZE3] / outputMass[i];
        const double X = inputMass[i] / outputMass[i];
        const double A = 1.0 / (1.0 + G);
        double B, BQ, E, D, C1, C;
        #pragma omp simd
        for(int j = 0; j < nOut; j++) {
            idx2 = i * nOut + j;

            BQ = 0.0;
            for(int k = 1; k < SIZE3; k++) {
                idx3 = idx2 * SIZE3 + k;

                B = outputMomSum[i * SIZE3 + k] / outputMass[i];
                BQ -= B * outputParticles[idx3];
            }

            E = outputParticles[idx2 * SIZE3];
            D = G * E + BQ;
            C1 = E + A * BQ;

            idx3 = idx2 * SIZE3;
            points[idx3] = X * D;
            for(int k = 1; k < SIZE3; k++) {
                idx3 = idx2 * SIZE3 + k;

                B = -outputMomSum[i * SIZE3 + k] / outputMass[i];
                C = outputParticles[idx3] + B * C1;
                points[idx3] = X * C;
            }
        }
    }

    return points;
}

void rambo(size_t nPoints) {
    constexpr int ecms = 100;
    constexpr int nOut = 4;

    auto e = generatePoints(ecms, nPoints, nOut);
    unique_ptr<double[]> h(new double[SIZE3 * nPoints]);

    for(int i = 0; i < SIZE3; i++) {
        #pragma omp parallel for
        for(size_t j = 0; j < nPoints; j++) {
            size_t idx2 = j * nOut;
            size_t idx3 = idx2 * SIZE3 + i;
            double max = e[idx3];
            double val;
            for(int k = 1; k < 3; k++) {
                idx2 = j * nOut + k;
                idx3 = idx2 * SIZE3 + i;
                val = e[idx3];
                if(val > max) {
                    max = val;
                }
            }
            h[i * nPoints + j] = max;
        }
    }
}
