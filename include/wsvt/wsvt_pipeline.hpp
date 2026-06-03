#pragma once

#include "wsvt/pyramid.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace wsvt {

struct SolverOutput {
    std::vector<float> displace_y;
    std::vector<float> displace_x;
    std::size_t h;
    std::size_t w;
    std::vector<float> dpc_y;
    std::vector<float> dpc_x;
    std::vector<float> phase;
    std::vector<float> transmission;
    std::vector<float> darkfield;
    std::vector<float> darkfield_nd;
    std::size_t transmission_h;
    std::size_t transmission_w;
    double time_cost_s = 0.0;
    double pyramid_time_s = 0.0;
    double wavelet_time_s = 0.0;
    double displace_time_s = 0.0;
    double postprocess_time_s = 0.0;
    double darkfield_time_s = 0.0;
    double load_time_s = 0.0;
    double save_time_s = 0.0;
};

class WSVT {
public:
    WSVT(
        const std::vector<float>& img_stack,
        const std::vector<float>& ref_stack,
        std::size_t ch,
        std::size_t h,
        std::size_t w,
        int crop = 512,
        int cal_half_window = 20,
        int n_template = 0,
        int n_s_extend = 4,
        int n_cores = 4,
        int n_group = 4,
        double energy = 14e3,
        double p_x = 0.65e-6,
        double mag_factor = 1.0,
        double z = 500e-3,
        int wavelet_level_cut = 2,
        int pyramid_level = 2,
        int n_iter = 1,
        bool use_estimate = false,
        bool use_wavelet = true,
        int use_gpu = 0,
        bool calc_darkfield = true);

    /// Move-semantics overload: takes ownership of img/ref data without copying
    WSVT(
        std::vector<float>&& img_stack,
        std::vector<float>&& ref_stack,
        std::size_t ch,
        std::size_t h,
        std::size_t w,
        int crop = 512,
        int cal_half_window = 20,
        int n_template = 0,
        int n_s_extend = 4,
        int n_cores = 4,
        int n_group = 4,
        double energy = 14e3,
        double p_x = 0.65e-6,
        double mag_factor = 1.0,
        double z = 500e-3,
        int wavelet_level_cut = 2,
        int pyramid_level = 2,
        int n_iter = 1,
        bool use_estimate = false,
        bool use_wavelet = true,
        int use_gpu = 0,
        bool calc_darkfield = true);

    AlignedVector<float> stack_TemplateWindow(const std::vector<float>& img, std::size_t in_ch, std::size_t in_h, std::size_t in_w, std::size_t& out_h, std::size_t& out_w, std::size_t& out_d) const;
    PyramidResult pyramid_data();
    PyramidResult wavelet_data();
    SolverOutput solver();
    SolverOutput run(const std::string& result_path = "", bool cleansave = false);

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
    std::size_t ch_;
    std::size_t h_;
    std::size_t w_;
    int crop_;
    int cal_half_window_;
    int n_s_extend_;
    int n_template_;
    int n_cores_;
    int n_group_;
    double mag_factor_;
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
    bool calc_darkfield_;
    int wavelet_level_;
    std::vector<int> wavelet_add_list_;
    std::vector<float> displace_estimate_y_;
    std::vector<float> displace_estimate_x_;
    std::size_t displace_estimate_h_;
    std::size_t displace_estimate_w_;
    double last_pyramid_time_s_;
    double last_wavelet_time_s_;
};

}
