// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>

using namespace sycl;

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

void pathfinder_impl(int iteration,
                     int64_t *gpuWall,
                     int64_t *gpuSrc,
                     int64_t *gpuResults,
                     int cols,
                     int rows,
                     int startStep,
                     int border,
                     sycl::nd_item<3> item_ct1,
                     int block_size,
                     int64_t *prev,
                     int64_t *result)
{

    int bx = item_ct1.get_group(2);
    int tx = item_ct1.get_local_id(2);

    // each block finally computes result for a small block
    // after N iterations.
    // it is the non-overlapping small blocks that cover
    // all the input data

    // calculate the small block size
    int small_block_cols = block_size - iteration * 2;

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + block_size - 1;

    // calculate the global thread coordination
    int xidx = blkX + tx;

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? block_size - 1 - (blkXmax - cols + 1)
                                         : block_size - 1;

    int W = tx - 1;
    int E = tx + 1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx, 0, cols - 1)) {
        prev[tx] = gpuSrc[xidx];
    }
    item_ct1.barrier(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    bool computed;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, block_size - i - 2) && isValid) {
            computed = true;
            int64_t left = prev[W];
            int64_t up = prev[tx];
            int64_t right = prev[E];
            int64_t shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            int index = cols * (startStep + i) + xidx;
            result[tx] = shortest + gpuWall[index];
        }
        item_ct1.barrier();
        if (i == iteration - 1)
            break;
        if (computed) // Assign the computation range
            prev[tx] = result[tx];
        item_ct1
            .barrier(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed) {
        gpuResults[xidx] = result[tx];
    }
}
