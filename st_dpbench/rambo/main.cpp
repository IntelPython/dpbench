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

// For M_PI
#define _USE_MATH_DEFINES
#include "data_gen.h"
#include "rdtsc.h"
#include <CL/sycl.hpp>
#include <math.h>
#include <stdlib.h>
#include <type_traits>

using namespace std;
using namespace cl::sycl;

template <typename FpTy> class RamboKernel;

template <typename FpTy>
void rambo_impl(queue &Queue,
                size_t nevts,
                size_t nout,
                const FpTy *usmC1,
                const FpTy *usmF1,
                const FpTy *usmQ1,
                FpTy *usmOutput)
{
    constexpr FpTy pi_v = M_PI;
    Queue.submit([&](handler &h) {
        h.parallel_for<RamboKernel<FpTy>>(range<1>{nevts}, [=](id<1> myID) {
            for (size_t j = 0; j < nout; j++) {
                int i = myID[0];
                size_t idx = i * nout + j;

                FpTy C = 2 * usmC1[idx] - 1;
                FpTy S = sycl::sqrt(1 - C * C);
                FpTy F = 2 * pi_v * usmF1[idx];
                FpTy Q = -sycl::log(usmQ1[idx]);

                usmOutput[idx * 4] = Q;
                usmOutput[idx * 4 + 1] = Q * S * sycl::sin(F);
                usmOutput[idx * 4 + 2] = Q * S * sycl::cos(F);
                usmOutput[idx * 4 + 3] = Q * C;
            }
        });
    });
    Queue.wait();
}

int main(int argc, char *argv[])
{
    size_t nevts = 16777216;
    size_t nout = 4;
    clock_t t1 = 0, t2 = 0;

    queue q;

    double *C1, *F1, *Q1, *output;

    /* Allocate arrays, generate input data */
    InitData(q, nevts, nout, &C1, &F1, &Q1, &output);

    /* Warm up cycle */
    rambo_impl<double>(q, nevts, nout, C1, F1, Q1, output);

    t1 = timer_rdtsc();
    rambo_impl<double>(q, nevts, nout, C1, F1, Q1, output);
    t2 = timer_rdtsc();

    printf("%.6lf\n", (double)(t2 - t1) / getHz());
    fflush(stdout);

    return 0;
}
