import base_gpairs
import numpy as np
import gaussian_weighted_pair_counts as gwpc
import dpctl, dpctl.tensor as dpt


def ceiling_quotient(n, m):
    return int((n + m - 1) / m)


def count_weighted_pairs_3d_intel_no_slm(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    n_wi = 20
    private_hist_size = 16
    lws0 = 16
    lws1 = 16

    m0 = n_wi * lws0
    m1 = n_wi * lws1

    n_groups0 = ceiling_quotient(n, m0)
    n_groups1 = ceiling_quotient(n, m1)

    gwsRange = n_groups0 * lws0, n_groups1 * lws1
    lwsRange = lws0, lws1

    slm_hist_size = ceiling_quotient(nbins, private_hist_size) * private_hist_size

    with dpctl.device_context(base_gpairs.get_device_selector()):
        gwpc.count_weighted_pairs_3d_intel_no_slm[gwsRange, lwsRange](
            n,
            nbins,
            slm_hist_size,
            private_hist_size,
            d_x1,
            d_y1,
            d_z1,
            d_w1,
            d_x2,
            d_y2,
            d_z2,
            d_w2,
            d_rbins_squared,
            d_result,
        )


def count_weighted_pairs_3d_intel_orig(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):

    # create tmp result on device
    result_tmp = np.zeros(nbins, dtype=np.float32)
    d_result_tmp = dpt.usm_ndarray(
        result_tmp.shape, dtype=result_tmp.dtype, buffer="device"
    )
    d_result_tmp.usm_data.copy_from_host(result_tmp.reshape((-1)).view("|u1"))

    with dpctl.device_context(base_gpairs.get_device_selector()):
        gwpc.count_weighted_pairs_3d_intel_orig_ker[n,](
            n,
            nbins,
            d_x1,
            d_y1,
            d_z1,
            d_w1,
            d_x2,
            d_y2,
            d_z2,
            d_w2,
            d_rbins_squared,
            d_result_tmp,
        )
        gwpc.count_weighted_pairs_3d_intel_agg_ker[
            nbins,
        ](d_result, d_result_tmp)


def run_gpairs(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    count_weighted_pairs_3d_intel_no_slm_ker(
        n,
        nbins,
        d_x1,
        d_y1,
        d_z1,
        d_w1,
        d_x2,
        d_y2,
        d_z2,
        d_w2,
        d_rbins_squared,
        d_result,
    )


base_gpairs.run("Gpairs Dppy kernel", run_gpairs)
