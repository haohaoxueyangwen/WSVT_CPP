#pragma once

#include "wsvt/pyramid.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace wsvt {

struct WXSTOutput {
    std::vector<float> displace_y;
    std::vector<float> displace_x;
    std::size_t h;
    std::size_t w;
    std::vector<float> dpc_y;
    std::vector<float> dpc_x;
    std::vector<float> phase;
    std::vector<float> transmission;
    std::size_t transmission_h;
    std::size_t transmission_w;
    std::vector<float> darkfield_nd;
    double time_cost_s = 0.0;
    double pyramid_time_s = 0.0;
    double wavelet_time_s = 0.0;
    double displace_time_s = 0.0;
    double postprocess_time_s = 0.0;
    double template_window_time_s = 0.0;
    double post_transmission_time_s = 0.0;
    double post_crop_sign_time_s = 0.0;
    double post_dpc_conv_time_s = 0.0;
    double post_phase_recovery_time_s = 0.0;
    double result_write_time_s = 0.0;
    double process_wall_s = 0.0;
    double load_time_s = 0.0;
    double save_time_s = 0.0;
};

class WXST {
public:
    WXST(
        const std::vector<float>& img,
        const std::vector<float>& ref,
        std::size_t h,
        std::size_t w,
        int m_image = 512,
        int n_s = 5,
        int cal_half_window = 20,
        int n_s_extend = 4,
        int n_cores = 4,
        int n_group = 4,
        double energy = 14e3,
        double p_x = 0.65e-6,
        double z = 500e-3,
        int wavelet_level_cut = 2,
        int pyramid_level = 2,
        int n_iter = 1,
        bool use_estimate = false,
        bool use_wavelet = true,
        int use_gpu = 0,
        int wavelet_impl = 0);

    PyramidResult pyramid_data();
    PyramidResult wavelet_data();
    WXSTOutput solver();
    WXSTOutput run(const std::string& result_path = "", int h5_deflate = 9);

private:
    std::vector<float> resampling_spline(const std::vector<float>& img, std::size_t in_h, std::size_t in_w, std::size_t out_h, std::size_t out_w) const;
    std::array<std::vector<float>, 3> displace_wavelet(
        std::span<const float> img_wa_stack,
        std::size_t img_h,
        std::size_t img_w,
        std::span<const float> ref_wa_stack,
        std::size_t ref_h,
        std::size_t ref_w,
        std::size_t depth,
        const std::vector<float>& displace_y,
        const std::vector<float>& displace_x,
        int cal_half_window,
        int n_pad) const;

    std::vector<float> img_data_;
    std::vector<float> ref_data_;
    std::size_t h_;
    std::size_t w_;
    int m_image_;
    int n_s_;
    int cal_half_window_;
    int n_s_extend_;
    int n_cores_;
    int n_group_;
    double energy_;
    double wavelength_;
    double p_x_;
    double z_;
    int wavelet_level_cut_;
    int pyramid_level_;
    int n_iter_;
    bool use_estimate_;
    bool use_wavelet_;
    bool use_gpu_;
    int wavelet_impl_;  // 0=streamed, 1=planned, 2=pixelchain (future)
    int wavelet_level_;
    std::vector<int> wavelet_add_list_;
    std::vector<float> displace_estimate_y_;
    std::vector<float> displace_estimate_x_;
    double last_pyramid_time_s_;
    double last_template_window_time_s_ = 0.0;
    double last_wavelet_time_s_;
};

}
