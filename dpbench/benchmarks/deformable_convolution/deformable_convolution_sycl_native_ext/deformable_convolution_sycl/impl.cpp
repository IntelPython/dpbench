//==- impl.cpp - Python native extension of deformable convolution   ===//
//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///
/// \file
/// The files implements a SYCL-based Python native extension for the
/// deformable convolution benchmark.

#include "utils.hpp"
#include "CL/sycl.hpp"
#include "dpctl4pybind11.hpp"
#include "oneapi/mkl.hpp"
#include "cmath"

using namespace sycl;
namespace py = pybind11;

template<class DataType>
__attribute__((always_inline)) DataType bilinear(const DataType* input,
                                                 int height,
                                                 int width,
                                                 float offset_y,
                                                 float offset_x)
{
    auto start_x = int(std::floor(offset_x));
    auto start_x_weight = 1 - (offset_x - start_x);
    auto start_y = int(std::floor(offset_y));
    auto start_y_weight = 1 - (offset_y - start_y);

    DataType result = 0;
    if (offset_x >= width || offset_y >= height || offset_x <= -1 || offset_y <= -1)
        return result;

    if (start_x >= 0 && start_y >= 0)
    {
        auto w0 = start_x_weight*start_y_weight;
        auto v0 = *get_ptr_2d(input, height, width, start_y, start_x);

        result += w0*v0;
    }

    if (start_x + 1 < width && start_y >= 0)
    {
        auto w1 = (1 - start_x_weight)*start_y_weight;
        auto v1 = *get_ptr_2d(input, height, width, start_y, start_x + 1);

        result += w1*v1;
    }

    if (start_x >=0 && start_y + 1 < height)
    {
        auto w2 = start_x_weight*(1 - start_y_weight);
        auto v2 = *get_ptr_2d(input, height, width, start_y + 1, start_x);

        result += w2*v2;
    }

    if (start_x + 1 < width && start_y + 1 < height)
    {
        auto w3 = (1 - start_x_weight)*(1 - start_y_weight);
        auto v3 = *get_ptr_2d(input, height, width, start_y + 1, start_x + 1);

        result += w3*v3;
    }

    return result;
}

class deform;

template<class DataType>
inline auto deform_input(cl::sycl::queue& queue,
                         const DataType* input, const Shape3D in_shape,
                         DataType* output, const Shape5D out_shape,
                         const DataType* offset,
                         int stride_y, int stride_x,
                         int pad_y, int pad_x,
                         int dilation_y, int dilation_x)
{
    auto in_channels = in_shape[CHW::C];
    auto in_height   = in_shape[CHW::H];
    auto in_width    = in_shape[CHW::W];

    auto k_height   = out_shape[CKHW::KH];
    auto k_width    = out_shape[CKHW::KW];
    auto out_height = out_shape[CKHW::H];
    auto out_width  = out_shape[CKHW::W];

    assert(out_shape[CKHW::C] == in_channels);

    auto wsize = sycl::range<3>(in_channels*k_height*k_width, out_height, out_width);
    return queue.parallel_for<deform>(wsize, [=](sycl::id<3> idx)
    {
        auto ckhkw = static_cast<int>(idx[0]);
        auto h = static_cast<int>(idx[1]);
        auto w = static_cast<int>(idx[2]);

        auto c = ckhkw/(k_height*k_width);
        auto khkw = ckhkw%(k_height*k_width);

        auto kh = khkw/k_width;
        auto kw = khkw%k_width;

        auto k_h_m = (k_height - 1)/2;
        auto k_w_m = (k_width - 1)/2;

        auto _output = get_ptr_5d(output,
                                  in_channels, k_height, k_width, out_height, out_width,
                                  c, kh, kw, h, 0);

        auto offset_y = *get_ptr_5d(offset, k_height, k_width, 2, out_height, out_width, kh, kw, 1, h, w) + h*stride_y + (kh - k_h_m)*dilation_y - (pad_y - k_h_m);
        auto offset_x = *get_ptr_5d(offset, k_height, k_width, 2, out_height, out_width, kh, kw, 0, h, w) + w*stride_x + (kw - k_w_m)*dilation_x - (pad_x - k_w_m);
        // auto offset_y = h*stride_y + (kh - k_h_m)*dilation_y - (pad_y - k_h_m);
        // auto offset_x = w*stride_x + (kw - k_w_m)*dilation_x - (pad_x - k_w_m);

        auto _input = get_ptr_3d(input, in_channels, in_height, in_width, c, 0, 0);

        _output[w] = bilinear(_input, in_height, in_width, offset_y, offset_x);
        // _output[w] = offset_y;
    });
}

class fill_output;

template<class DataType>
auto output_fill_with_bias(cl::sycl::queue& queue, DataType* output, const Shape3D out_shape, DataType* bias)
{
    auto out_c = out_shape[CHW::C];
    auto out_h = out_shape[CHW::H];
    auto out_w = out_shape[CHW::W];

    return queue.parallel_for<fill_output>(sycl::range<3>(out_c, out_h, out_w),[=](sycl::id<3> idx)
    {
        auto c = static_cast<int>(idx[0]);
        auto h = static_cast<int>(idx[1]);
        auto w = static_cast<int>(idx[2]);

        auto out_ptr = get_ptr_3d(output, out_c, out_h, out_w, c, h, w);
        *out_ptr = bias[c];
    });
}

template<class DataType>
void deformable_convolution_b1_impl(cl::sycl::queue& queue,
                                    const DataType* input,
                                    const Shape3D in_shape,
                                    DataType* output,
                                    const Shape3D out_shape,
                                    DataType* tmp,
                                    const float* offset,
                                    DataType* weights,
                                    const Shape4D weights_shape,
                                    DataType* bias,
                                    int stride_y, int stride_x,
                                    int pad_y, int pad_x,
                                    int dilation_y, int dilation_x,
                                    int groups, int deformable_groups)
{
    using oneapi::mkl::blas::row_major::gemm;
    using oneapi::mkl::transpose;

    auto in_c = in_shape[CHW::C];
    auto in_h = in_shape[CHW::H];
    auto in_w = in_shape[CHW::W];

    auto out_c = out_shape[CHW::C];
    auto out_h = out_shape[CHW::H];
    auto out_w = out_shape[CHW::W];

    assert(out_c == weights_shape[OIHW::OC]);
    assert(in_c == weights_shape[OIHW::IC]);
    auto ker_h = weights_shape[OIHW::H];
    auto ker_w = weights_shape[OIHW::W];

    auto edeform = deform_input(queue,
                                input, in_shape,
                                tmp, {in_c, ker_h, ker_w, out_h, out_w},
                                offset,
                                stride_y, stride_x,
                                pad_y, pad_x,
                                dilation_y, dilation_x);

    auto efill = output_fill_with_bias(queue, output, out_shape, bias);
    auto egemm = gemm(queue,
                      transpose::N, transpose::N, /*transpose a, b*/
                      out_c, out_h*out_w, in_c*ker_h*ker_w, /*m, n, k*/
                      1, /*alpha*/
                      weights, in_c*ker_h*ker_w, /*a, lda*/
                      tmp, out_h*out_w, /*b, ldb*/
                      1, /*beta*/
                      output, out_h*out_w, /*c, ldc*/
                      {edeform, efill} /*events*/);
    egemm.wait();
}

template<class DataType>
void deformable_convolution_impl(cl::sycl::queue& queue,
                                 const DataType* input,
                                 const Shape4D in_shape,
                                 DataType* output,
                                 const Shape4D out_shape,
                                 DataType* tmp,
                                 const float* offset,
                                 DataType* weights,
                                 const Shape4D weights_shape,
                                 DataType* bias,
                                 int stride_y, int stride_x,
                                 int pad_y, int pad_x,
                                 int dilation_y, int dilation_x,
                                 int groups, int deformable_groups)
{
    auto in_b = in_shape[NCHW::N];
    auto in_c = in_shape[NCHW::C];
    auto in_h = in_shape[NCHW::H];
    auto in_w = in_shape[NCHW::W];

    assert(in_b == out_shape[NCHW::N]);
    auto out_c = out_shape[NCHW::C];
    auto out_h = out_shape[NCHW::H];
    auto out_w = out_shape[NCHW::W];

    assert(out_c == weights_shape[OIHW::OC]);
    assert(in_c == weights_shape[OIHW::IC]);

    for (auto b = 0; b < in_b; ++b)
    {
        auto input_ptr = get_ptr_4d(input, in_b, in_c, in_h, in_w, b, 0, 0, 0);
        auto output_ptr = get_ptr_4d(output, in_b, out_c, out_h, out_w, b, 0, 0, 0);
        deformable_convolution_b1_impl(queue,
                                       input_ptr, {in_c, in_h, in_w},
                                       output_ptr, {out_c, out_h, out_w},
                                       tmp,
                                       offset,
                                       weights, weights_shape,
                                       bias,
                                       stride_y, stride_x,
                                       pad_y, pad_x,
                                       dilation_y, dilation_x,
                                       groups, deformable_groups);

    }
}

void deformable_convolution(dpctl::tensor::usm_ndarray input, dpctl::tensor::usm_ndarray output, dpctl::tensor::usm_ndarray offset,
                            dpctl::tensor::usm_ndarray weights, dpctl::tensor::usm_ndarray bias, dpctl::tensor::usm_ndarray tmp,
                            int stride_y, int stride_x, int pad_y, int pad_x, int dilation_y, int dilation_x, int groups, int deformable_groups)
{
    auto queue = input.get_queue();

    if (input.get_typenum() != UAR_FLOAT) {
        throw std::runtime_error("Expected a single precision FP array.");
    }

    int batch = input.get_shape(0);

    int in_channels = input.get_shape(1);
    int in_height = input.get_shape(2);
    int in_width = input.get_shape(3);

    int out_channels = output.get_shape(1);
    int out_height = output.get_shape(2);
    int out_width = output.get_shape(3);

    int kernel_height = weights.get_shape(2);
    int kernel_width = weights.get_shape(3);

    auto input_shape = Shape4D({batch, in_channels, in_height, in_width});
    auto output_shape = Shape4D({batch, out_channels, out_height, out_width});
    auto weights_shape = Shape4D({out_channels, in_channels, kernel_height, kernel_width});

    deformable_convolution_impl(queue,
                                input.get_data<float>(), input_shape,
                                output.get_data<float>(), output_shape,
                                tmp.get_data<float>(),
                                offset.get_data<float>(),
                                weights.get_data<float>(), weights_shape,
                                bias.get_data<float>(),
                                stride_y, stride_x,
                                pad_y, pad_x,
                                dilation_y, dilation_x,
                                groups, deformable_groups);
}

PYBIND11_MODULE(_deformable_convolution_sycl, m)
{
    import_dpctl();

    m.def("deformable_convolution", &deformable_convolution,
          "Defromable convolution",
          py::arg("input"), py::arg("output"), py::arg("offset"),
          py::arg("weights"), py::arg("bias"), py::arg("tmp"),
          py::arg("stride_y"), py::arg("stride_x"), py::arg("pad_y"), py::arg("pad_x"),
          py::arg("dilation_y"), py::arg("dilation_x"),
          py::arg("groups"), py::arg("deformable_groups"));
}
