/*
 * G-pairs algorithm:
 *   Input: n 3D points, given as x, y, and z vectors
 *          and associate vector of weights w
 *          nbins distance squared thresholds, as rbins vector
 *   Output:
 *          Vector of size nbins with computing
 *            hist[k] == sum( w[i1] * w[i2] *
 *                           bool( dist_sq(i1, i2) <= rbins[k] )
 *                       , [ 0<=i1<n, 0<=i2<n ])
 */

/*
 *  Description of the algorithm implementation
 *
 *  Each work-item processes n_wi points, work-group size is lws.
 *      n_groups = ceiling_quotient(n, n_wi * lws)
 *  Each group has local hist result accumulated over its local chunk
 *  of points.
 *
 *  Group histograms are added together as a post-processing step.
 */

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include "gpairs.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//#include "dpctl_sycl_types.h"
#include "dpctl4pybind11.hpp"
//#include "../_sycl_queue_api.h"
//#include "../_sycl_queue.h"

#ifdef __DO_FLOAT__
    typedef float tfloat;
#else
    typedef double tfloat;
#endif

namespace py = pybind11;

/* Python wrapper for gpairs_usm */
std::pair<py::array_t<tfloat, py::array::c_style>, double> py_gpairs(
								     //    sycl::queue q,
    py::array_t<tfloat, py::array::c_style> x0_arr,
    py::array_t<tfloat, py::array::c_style> y0_arr,
    py::array_t<tfloat, py::array::c_style> z0_arr,
    py::array_t<tfloat, py::array::c_style> w0_arr,
    py::array_t<tfloat, py::array::c_style> x1_arr,
    py::array_t<tfloat, py::array::c_style> y1_arr,
    py::array_t<tfloat, py::array::c_style> z1_arr,
    py::array_t<tfloat, py::array::c_style> w1_arr,
    py::array_t<tfloat, py::array::c_style> rbins_arr)
{
    sycl::queue q;
    py::buffer_info x0_pybuf = x0_arr.request();
    py::buffer_info y0_pybuf = y0_arr.request();
    py::buffer_info z0_pybuf = z0_arr.request();
    py::buffer_info w0_pybuf = w0_arr.request();

    py::buffer_info x1_pybuf = x1_arr.request();
    py::buffer_info y1_pybuf = y1_arr.request();
    py::buffer_info z1_pybuf = z1_arr.request();
    py::buffer_info w1_pybuf = w1_arr.request();

    py::buffer_info rbins_pybuf = rbins_arr.request();
    size_t n = x0_pybuf.size;
    size_t nbins = rbins_pybuf.size;
    if ((x0_pybuf.ndim == 1 && y0_pybuf.ndim == 1 &&
         z0_pybuf.ndim == 1 && w0_pybuf.ndim == 1 &&
         x1_pybuf.ndim == 1 && y1_pybuf.ndim == 1 &&
         z1_pybuf.ndim == 1 && w1_pybuf.ndim == 1 &&
         rbins_pybuf.ndim == 1))
    {
        if (!(n == y0_pybuf.size && n == z0_pybuf.size && n == w0_pybuf.size &&
              n == x1_pybuf.size && n == y1_pybuf.size && n == z1_pybuf.size &&
              n == w1_pybuf.size))
        {
            throw std::runtime_error("Expecting x, y, z, w to be vectors of the same length");
        }
    }
    else
    {
        throw std::runtime_error("Expecting inputs to be vectors");
    }

    tfloat *x0 = static_cast<tfloat *>(x0_pybuf.ptr);
    tfloat *y0 = static_cast<tfloat *>(y0_pybuf.ptr);
    tfloat *z0 = static_cast<tfloat *>(z0_pybuf.ptr);
    tfloat *w0 = static_cast<tfloat *>(w0_pybuf.ptr);

    tfloat *x1 = static_cast<tfloat *>(x1_pybuf.ptr);
    tfloat *y1 = static_cast<tfloat *>(y1_pybuf.ptr);
    tfloat *z1 = static_cast<tfloat *>(z1_pybuf.ptr);
    tfloat *w1 = static_cast<tfloat *>(w1_pybuf.ptr);

    tfloat *rbins = static_cast<tfloat *>(rbins_pybuf.ptr);


    tfloat *x0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_x0 = q.copy(/*src */ x0, /*dest */ x0_usm, n);

    tfloat *x1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_x1 = q.copy(/*src */ x1, /*dest */ x1_usm, n);

    tfloat *y0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_y0 = q.copy(y0, y0_usm, n);

    tfloat *y1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_y1 = q.copy(y1, y1_usm, n);

    tfloat *z0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_z0 = q.copy(z0, z0_usm, n);

    tfloat *z1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_z1 = q.copy(z1, z1_usm, n);

    tfloat *w0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_w0 = q.copy(w0, w0_usm, n);

    tfloat *w1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_w1 = q.copy(w1, w1_usm, n);

    tfloat *rbins_usm = sycl::malloc_device<tfloat>(nbins, q);
    sycl::event cp_rbins = q.copy(rbins, rbins_usm, nbins);

    tfloat *hist_usm = sycl::malloc_device<tfloat>(nbins, q);
    sycl::event init_hist_ev = q.fill<tfloat>(hist_usm, tfloat(0), nbins);

    sycl::kernel_id gpairs_kernel_id =
        sycl::get_kernel_id<eff_gpairs_kernel>();

    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        q.get_context(), {gpairs_kernel_id});

    sycl::kernel eff_gpairs_kern_obj = kb.get_kernel(gpairs_kernel_id);

    size_t private_mem_size = eff_gpairs_kern_obj.get_info<
        sycl::info::kernel_device_specific::private_mem_size>(q.get_device());

    std::cout << "Kernel's private mem size is: " << private_mem_size << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    sycl::event gpairs_ev = gpairs_usm<tfloat, tfloat>(
        q, n,
        x0_usm, y0_usm, z0_usm, w0_usm,
        x1_usm, y1_usm, z1_usm, w1_usm,
        nbins, rbins_usm, hist_usm,
        {cp_x0, cp_y0, cp_z0, cp_w0,
         cp_x1, cp_y1, cp_z1, cp_w1,
         cp_rbins, init_hist_ev});
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    auto totTime=std::chrono::duration_cast<std::chrono::microseconds>(diff).count();

  
    auto hist_arr = py::array_t<tfloat>(nbins);
    py::buffer_info hist_pybuf = hist_arr.request();
    tfloat *hist = static_cast<tfloat *>(hist_pybuf.ptr);

    q.copy<tfloat>(hist_usm, hist, nbins, {gpairs_ev}).wait();

    sycl::free(x0_usm, q);
    sycl::free(x1_usm, q);

    sycl::free(y0_usm, q);
    sycl::free(y1_usm, q);

    sycl::free(z0_usm, q);
    sycl::free(z1_usm, q);

    sycl::free(w0_usm, q);
    sycl::free(w1_usm, q);

    sycl::free(rbins_usm, q);
    sycl::free(hist_usm, q);

    return {hist_arr, totTime};
}

py::array_t<tfloat, py::array::c_style> py_gpairs_host(
    py::array_t<tfloat, py::array::c_style> x0_arr,
    py::array_t<tfloat, py::array::c_style> y0_arr,
    py::array_t<tfloat, py::array::c_style> z0_arr,
    py::array_t<tfloat, py::array::c_style> w0_arr,
    py::array_t<tfloat, py::array::c_style> x1_arr,
    py::array_t<tfloat, py::array::c_style> y1_arr,
    py::array_t<tfloat, py::array::c_style> z1_arr,
    py::array_t<tfloat, py::array::c_style> w1_arr,
    py::array_t<tfloat, py::array::c_style> rbins_arr)
{

    py::buffer_info x0_pybuf = x0_arr.request();
    py::buffer_info y0_pybuf = y0_arr.request();
    py::buffer_info z0_pybuf = z0_arr.request();
    py::buffer_info w0_pybuf = w0_arr.request();

    py::buffer_info x1_pybuf = x1_arr.request();
    py::buffer_info y1_pybuf = y1_arr.request();
    py::buffer_info z1_pybuf = z1_arr.request();
    py::buffer_info w1_pybuf = w1_arr.request();

    py::buffer_info rbins_pybuf = rbins_arr.request();

    size_t n = x0_pybuf.size;
    size_t nbins = rbins_pybuf.size;
    if ((x0_pybuf.ndim == 1 && y0_pybuf.ndim == 1 && z0_pybuf.ndim == 1 && w0_pybuf.ndim == 1 &&
         x1_pybuf.ndim == 1 && y1_pybuf.ndim == 1 && z1_pybuf.ndim == 1 && w1_pybuf.ndim == 1 &&
         rbins_pybuf.ndim == 1))
    {
        if (!(n == y0_pybuf.size && n == z0_pybuf.size && n == w0_pybuf.size &&
              n == x1_pybuf.size && n == y1_pybuf.size && n == z1_pybuf.size && n == w1_pybuf.size))
        {
            throw std::runtime_error("Expecting x, y, z, w to be vectors of the same length");
        }
    }
    else
    {
        throw std::runtime_error("Expecting inputs to be vectors");
    }

    tfloat *x0 = static_cast<tfloat *>(x0_pybuf.ptr);
    tfloat *y0 = static_cast<tfloat *>(y0_pybuf.ptr);
    tfloat *z0 = static_cast<tfloat *>(z0_pybuf.ptr);
    tfloat *w0 = static_cast<tfloat *>(w0_pybuf.ptr);

    tfloat *x1 = static_cast<tfloat *>(x1_pybuf.ptr);
    tfloat *y1 = static_cast<tfloat *>(y1_pybuf.ptr);
    tfloat *z1 = static_cast<tfloat *>(z1_pybuf.ptr);
    tfloat *w1 = static_cast<tfloat *>(w1_pybuf.ptr);

    tfloat *rbins = static_cast<tfloat *>(rbins_pybuf.ptr);

    auto hist_arr = py::array_t<tfloat>(nbins);
    py::buffer_info hist_pybuf = hist_arr.request();
    tfloat *hist = static_cast<tfloat *>(hist_pybuf.ptr);

    for (size_t i = 0; i < nbins; ++i)
    {
        hist[i] = tfloat(0);
    }

    gpairs_host_data_naive<tfloat, tfloat>(
        n,
        x0, y0, z0, w0,
        x1, y1, z1, w1,
        nbins, rbins, hist);

    return hist_arr;
}

#if 0
/* Python wrapper for gpairs_usm */
py::array_t<tfloat, py::array::c_style> py_gpairs_dpbench(
    py::sycl queue,
    py::array_t<tfloat, py::array::c_style> x0_arr,
    py::array_t<tfloat, py::array::c_style> y0_arr,
    py::array_t<tfloat, py::array::c_style> z0_arr,
    py::array_t<tfloat, py::array::c_style> w0_arr,
    py::array_t<tfloat, py::array::c_style> x1_arr,
    py::array_t<tfloat, py::array::c_style> y1_arr,
    py::array_t<tfloat, py::array::c_style> z1_arr,
    py::array_t<tfloat, py::array::c_style> w1_arr,
    py::array_t<tfloat, py::array::c_style> rbins_arr)
{

    py::buffer_info x0_pybuf = x0_arr.request();
    py::buffer_info y0_pybuf = y0_arr.request();
    py::buffer_info z0_pybuf = z0_arr.request();
    py::buffer_info w0_pybuf = w0_arr.request();

    py::buffer_info x1_pybuf = x1_arr.request();
    py::buffer_info y1_pybuf = y1_arr.request();
    py::buffer_info z1_pybuf = z1_arr.request();
    py::buffer_info w1_pybuf = w1_arr.request();

    py::buffer_info rbins_pybuf = rbins_arr.request();
    size_t n = x0_pybuf.size;
    size_t nbins = rbins_pybuf.size;
    if ((x0_pybuf.ndim == 1 && y0_pybuf.ndim == 1 &&
         z0_pybuf.ndim == 1 && w0_pybuf.ndim == 1 &&
         x1_pybuf.ndim == 1 && y1_pybuf.ndim == 1 &&
         z1_pybuf.ndim == 1 && w1_pybuf.ndim == 1 &&
         rbins_pybuf.ndim == 1))
    {
        if (!(n == y0_pybuf.size && n == z0_pybuf.size && n == w0_pybuf.size &&
              n == x1_pybuf.size && n == y1_pybuf.size && n == z1_pybuf.size &&
              n == w1_pybuf.size))
        {
            throw std::runtime_error("Expecting x, y, z, w to be vectors of the same length");
        }
    }
    else
    {
        throw std::runtime_error("Expecting inputs to be vectors");
    }

    tfloat *x0 = static_cast<tfloat *>(x0_pybuf.ptr);
    tfloat *y0 = static_cast<tfloat *>(y0_pybuf.ptr);
    tfloat *z0 = static_cast<tfloat *>(z0_pybuf.ptr);
    tfloat *w0 = static_cast<tfloat *>(w0_pybuf.ptr);

    tfloat *x1 = static_cast<tfloat *>(x1_pybuf.ptr);
    tfloat *y1 = static_cast<tfloat *>(y1_pybuf.ptr);
    tfloat *z1 = static_cast<tfloat *>(z1_pybuf.ptr);
    tfloat *w1 = static_cast<tfloat *>(w1_pybuf.ptr);

    tfloat *rbins = static_cast<tfloat *>(rbins_pybuf.ptr);

    auto q(*q_ptr);

    tfloat *x0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_x0 = q.copy(/*src */ x0, /*dest */ x0_usm, n);

    tfloat *x1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_x1 = q.copy(/*src */ x1, /*dest */ x1_usm, n);

    tfloat *y0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_y0 = q.copy(y0, y0_usm, n);

    tfloat *y1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_y1 = q.copy(y1, y1_usm, n);

    tfloat *z0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_z0 = q.copy(z0, z0_usm, n);

    tfloat *z1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_z1 = q.copy(z1, z1_usm, n);

    tfloat *w0_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_w0 = q.copy(w0, w0_usm, n);

    tfloat *w1_usm = sycl::malloc_device<tfloat>(n, q);
    sycl::event cp_w1 = q.copy(w1, w1_usm, n);

    tfloat *rbins_usm = sycl::malloc_device<tfloat>(nbins, q);
    sycl::event cp_rbins = q.copy(rbins, rbins_usm, nbins);

    tfloat *hist_usm = sycl::malloc_device<tfloat>(nbins, q);
    sycl::event init_hist_ev = q.fill<tfloat>(hist_usm, tfloat(0), nbins);

    sycl::event gpairs_ev = gpairs_dpbench<tfloat, tfloat>(
        q, n,
        x0_usm, y0_usm, z0_usm, w0_usm,
        x1_usm, y1_usm, z1_usm, w1_usm,
        nbins, rbins_usm, hist_usm,
        {cp_x0, cp_y0, cp_z0, cp_w0,
         cp_x1, cp_y1, cp_z1, cp_w1,
         cp_rbins, init_hist_ev});

    auto hist_arr = py::array_t<tfloat>(nbins);
    py::buffer_info hist_pybuf = hist_arr.request();
    tfloat *hist = static_cast<tfloat *>(hist_pybuf.ptr);

    q.copy<tfloat>(hist_usm, hist, nbins, {gpairs_ev}).wait();

    sycl::free(x0_usm, q);
    sycl::free(x1_usm, q);
    sycl::free(y0_usm, q);
    sycl::free(y1_usm, q);
    sycl::free(z0_usm, q);
    sycl::free(z1_usm, q);
    sycl::free(w0_usm, q);
    sycl::free(w1_usm, q);
    sycl::free(rbins_usm, q);
    sycl::free(hist_usm, q);

    return hist_arr;
}
#endif

PYBIND11_MODULE(sycl_gpairs, m)
{
    import_dpctl();
    m.def("sycl_gpairs",
          &py_gpairs,
          "Evaluate g-pairs algorithm on SYCL queue");
    m.def("host_gpairs",
          &py_gpairs_host,
          "Evaluate g-pairs algorithm on host");
    // m.def("dpbench_gpairs",
    //       &py_gpairs_dpbench,
    //       "Evaluate g-pairs algorithms using dpbench implementation");
}
