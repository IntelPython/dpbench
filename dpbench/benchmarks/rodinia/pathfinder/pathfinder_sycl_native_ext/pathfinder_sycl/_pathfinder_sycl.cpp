// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_pathfinder_kernel.hpp"
#include <CL/sycl.hpp>
#include <dpctl4pybind11.hpp>

using namespace sycl;

template <typename... Args> bool ensure_compatibility(const Args &...args)
{
    std::vector<dpctl::tensor::usm_ndarray> arrays = {args...};

    auto arr = arrays.at(0);
    auto q = arr.get_queue();
    auto type_flag = arr.get_typenum();
    auto arr_size = arr.get_size();

    for (auto &arr : arrays) {
        if (!(arr.get_flags() & (USM_ARRAY_C_CONTIGUOUS))) {
            std::cerr << "All arrays need to be C contiguous.\n";
            return false;
        }
        if (arr.get_typenum() != type_flag) {
            std::cerr << "All arrays should be of same elemental type.\n";
            return false;
        }
        if (arr.get_ndim() > 1) {
            std::cerr << "All arrays expected to be single-dimensional.\n";
            return false;
        }
    }
    return true;
}

void pathfinder_sync(dpctl::tensor::usm_ndarray data,
                     int rows,
                     int cols,
                     int pyramid_height,
                     int block_size,
                     dpctl::tensor::usm_ndarray result)
{

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height);
    int smallBlockCol = block_size - (pyramid_height)*2;
    int blockCols =
        cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    int64_t *gpuWall, *gpuResult[2];
    int size = rows * cols;

    auto defaultQueue = data.get_queue();

    if (!ensure_compatibility(data, result))
        throw std::runtime_error("Input arrays are not acceptable.");

    gpuResult[0] = sycl::malloc_device<int64_t>(cols, defaultQueue);
    gpuResult[1] = sycl::malloc_device<int64_t>(cols, defaultQueue);
    gpuWall = sycl::malloc_device<int64_t>((size - cols), defaultQueue);

    // Extract value ptr from dpctl array
    int64_t *data_value = data.get_data<int64_t>();

    defaultQueue.memcpy(gpuResult[0], data_value, sizeof(int64_t) * cols)
        .wait();

    defaultQueue
        .memcpy(gpuWall, data_value + cols, sizeof(int64_t) * (size - cols))
        .wait();

    sycl::range<3> dimBlock(1, 1, block_size);
    sycl::range<3> dimGrid(1, 1, blockCols);
    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        /*
        DPCT1049:0: The workgroup size passed to the SYCL
         * kernel may exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
        */
        defaultQueue
            .submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int64_t, 1> prev_acc_ct1(
                    sycl::range<1>(256 /*block_size*/), cgh);
                sycl::local_accessor<int64_t, 1> result_acc_ct1(
                    sycl::range<1>(256 /*block_size*/), cgh);
                auto gpuResult_src_ct2 = gpuResult[src];
                auto gpuResult_dst_ct3 = gpuResult[dst];
                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        pathfinder_impl(MIN(pyramid_height, rows - t - 1),
                                        gpuWall, gpuResult_src_ct2,
                                        gpuResult_dst_ct3, cols, rows, t,
                                        borderCols, item_ct1, block_size,
                                        prev_acc_ct1.get_pointer(),
                                        result_acc_ct1.get_pointer());
                    });
            })
            .wait();
    }

    // Extract value for result ptr
    auto result_value = result.get_data<int64_t>();

    defaultQueue.memcpy(result_value, gpuResult[dst], sizeof(int64_t) * cols)
        .wait();

    sycl::free(gpuWall, defaultQueue);
    sycl::free(gpuResult[0], defaultQueue);
    sycl::free(gpuResult[1], defaultQueue);
}

PYBIND11_MODULE(_pathfinder_sycl, m)
{
    // Import the dpctl extensions
    import_dpctl();

    m.def("pathfinder", &pathfinder_sync,
          "DPC++ implementation of the pathfinder", py::arg("data"),
          py::arg("rows"), py::arg("cols"), py::arg("pyramid_height"),
          py::arg("block_size"), py::arg("result"));
}
