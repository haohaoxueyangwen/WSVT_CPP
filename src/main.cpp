#include "wsvt/io_h5.hpp"
#include "wsvt/io_image.hpp"
#include "wsvt/wsvt_pipeline.hpp"
#include "wsvt/wxst_pipeline.hpp"

#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration to avoid Image2D template conflict between
// io_image.hpp and image.hpp (included transitively via align_ops.hpp)
namespace wsvt {
struct StackAlignResult {
    std::array<double, 2> shift;
    double error;
    double diffphase;
};
StackAlignResult stack_image_align(
    std::vector<float>& ref_stack,
    const float* img_first_frame, std::size_t img_stride,
    std::size_t ch, std::size_t h, std::size_t w);
}

namespace {

using OptMap = std::unordered_map<std::string, std::string>;

OptMap parse_opts(int argc, char** argv, int start) {
    OptMap opts;
    if ((argc - start) % 2 != 0) {
        throw std::invalid_argument("options must be key-value pairs");
    }
    for (int i = start; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        if (key.rfind("--", 0) == 0) {
            key = key.substr(2);
        }
        opts[key] = argv[i + 1];
    }
    return opts;
}

int opt_int(const OptMap& opts, const std::string& key, int def) {
    const auto it = opts.find(key);
    if (it == opts.end()) return def;
    return std::stoi(it->second);
}

double opt_double(const OptMap& opts, const std::string& key, double def) {
    const auto it = opts.find(key);
    if (it == opts.end()) return def;
    return std::stod(it->second);
}

bool opt_bool(const OptMap& opts, const std::string& key, bool def) {
    const auto it = opts.find(key);
    if (it == opts.end()) return def;
    std::string v = it->second;
    for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "on" || v == "yes") return true;
    if (v == "0" || v == "false" || v == "off" || v == "no") return false;
    throw std::invalid_argument("invalid bool option: " + key);
}

int run_demo() {
    std::vector<float> wxst_img(32 * 32, 0.0f);
    std::vector<float> wxst_ref(32 * 32, 0.0f);
    for (std::size_t y = 0; y < 32; ++y) {
        for (std::size_t x = 0; x < 32; ++x) {
            wxst_ref[y * 32 + x] = static_cast<float>((y + 1) * (x + 1));
            wxst_img[y * 32 + x] = wxst_ref[y * 32 + ((x + 1) % 32)];
        }
    }
    wsvt::WXST wxst(wxst_img, wxst_ref, 32, 32, 32, 1, 2, 2, 2, 2, 14000.0, 0.65e-6, 0.5, 2, 1, 1, false, true, 0);
    const auto wxst_sol = wxst.solver();
    std::cout << "wxst_displace_shape=" << wxst_sol.h << "x" << wxst_sol.w << std::endl;
    return 0;
}

int run_wxst_cmd(const std::string& img_h5, const std::string& img_key, const std::string& ref_h5, const std::string& ref_key, const std::string& out_dir, const OptMap& opts) {
    const auto t_process_t0 = std::chrono::steady_clock::now();
    const auto t_load_t0 = std::chrono::steady_clock::now();
    const auto img = wsvt::read_h5(img_h5, img_key, false);
    const auto ref = wsvt::read_h5(ref_h5, ref_key, false);
    if (img.shape.size() != 2 || ref.shape.size() != 2) {
        throw std::invalid_argument("wxst input must be 2D");
    }
    if (img.shape != ref.shape) {
        throw std::invalid_argument("wxst img/ref shape mismatch");
    }
    const auto t_load_t1 = std::chrono::steady_clock::now();
    const double load_time_s = std::chrono::duration<double>(t_load_t1 - t_load_t0).count();
    const std::size_t h = img.shape[0];
    const std::size_t w = img.shape[1];
    const int m_image = opt_int(opts, "m_image", static_cast<int>(h));
    const int n_s = opt_int(opts, "n_s", 5);
    const int cal_half_window = opt_int(opts, "cal_half_window", 20);
    const int n_s_extend = opt_int(opts, "n_s_extend", 4);
    const int n_cores = opt_int(opts, "n_cores", 4);
    const int n_group = opt_int(opts, "n_group", 4);
    const double energy = opt_double(opts, "energy", 14000.0);
    const double p_x = opt_double(opts, "p_x", 0.65e-6);
    const double z = opt_double(opts, "z", 0.5);
    const int wavelet_level_cut = opt_int(opts, "wavelet_level_cut", 2);
    const int pyramid_level = opt_int(opts, "pyramid_level", 2);
    const int n_iter = opt_int(opts, "n_iter", 1);
    const bool use_estimate = opt_bool(opts, "use_estimate", false);
    const bool use_wavelet = opt_bool(opts, "use_wavelet", true);
    const int use_gpu = opt_bool(opts, "use_gpu", false) ? 1 : 0;
    const int wavelet_impl = opt_int(opts, "wavelet_impl", 0);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WXST wxst(
        img.data, ref.data, h, w,
        m_image, n_s, cal_half_window, n_s_extend, n_cores, n_group,
        energy, p_x, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, wavelet_impl);
    const auto out = wxst.run(out_dir, h5_deflate);
    const auto t_save_t0 = std::chrono::steady_clock::now();
    if (save_img) {
        const std::filesystem::path od(out_dir);
        wsvt::save_img(out.displace_x, out.h, out.w, (od / "displace_x.tif").string());
        wsvt::save_img(out.displace_y, out.h, out.w, (od / "displace_y.tif").string());
        wsvt::save_img(out.dpc_x, out.h, out.w, (od / "DPC_x.tif").string());
        wsvt::save_img(out.dpc_y, out.h, out.w, (od / "DPC_y.tif").string());
        wsvt::save_img(out.phase, out.h, out.w, (od / "phase.tif").string());
        wsvt::save_img(out.transmission, out.transmission_h, out.transmission_w, (od / "transmission_image.tif").string());
        wsvt::save_img(out.darkfield_nd, out.h, out.w, (od / "darkfield_nd.tif").string());
    }
    const auto t_save_t1 = std::chrono::steady_clock::now();
    const double save_time_s = std::chrono::duration<double>(t_save_t1 - t_save_t0).count();
    const auto t_process_t1 = std::chrono::steady_clock::now();
    const double process_wall_s = std::chrono::duration<double>(t_process_t1 - t_process_t0).count();
    std::cout << "=== Timing Summary ===" << std::endl;
    std::cout << "  load:             " << load_time_s << " s" << std::endl;
    std::cout << "  pyramid:          " << out.pyramid_time_s << " s" << std::endl;
    std::cout << "  template_window:  " << out.template_window_time_s << " s" << std::endl;
    std::cout << "  wavelet:          " << out.wavelet_time_s << " s" << std::endl;
    std::cout << "  displace:         " << out.displace_time_s << " s" << std::endl;
    std::cout << "  post-proc:        " << out.postprocess_time_s << " s" << std::endl;
    std::cout << "  result write:     " << out.result_write_time_s << " s"
              << " (h5_deflate=" << h5_deflate << ")" << std::endl;
    std::cout << "  optional_image_save: " << save_time_s << " s" << std::endl;
    std::cout << "  solver:           " << out.time_cost_s << " s" << std::endl;
    std::cout << "  wall total:       " << (load_time_s + out.time_cost_s + out.result_write_time_s + save_time_s) << " s" << std::endl;
    std::cout << "  process wall:     " << process_wall_s << " s" << std::endl;
    std::cout << "wxst done: " << out.h << "x" << out.w << std::endl;
    return 0;
}

int run_wxst_dir_cmd(const std::string& img_dir, const std::string& ref_dir, const std::string& out_dir, const OptMap& opts) {
    const auto t_process_t0 = std::chrono::steady_clock::now();
    const auto t_load_t0 = std::chrono::steady_clock::now();
    const auto img_files = wsvt::list_image_files(img_dir);
    const auto ref_files = wsvt::list_image_files(ref_dir);
    if (img_files.size() != 1 || ref_files.size() != 1) {
        throw std::invalid_argument("wxst_dir expects exactly 1 image in each folder");
    }
    const auto img = wsvt::read_image_gray(img_files[0]);
    const auto ref = wsvt::read_image_gray(ref_files[0]);
    if (img.h != ref.h || img.w != ref.w) {
        throw std::invalid_argument("wxst_dir img/ref shape mismatch");
    }
    const auto t_load_t1 = std::chrono::steady_clock::now();
    const double load_time_s = std::chrono::duration<double>(t_load_t1 - t_load_t0).count();
    const int m_image = opt_int(opts, "m_image", static_cast<int>(img.h));
    const int n_s = opt_int(opts, "n_s", 5);
    const int cal_half_window = opt_int(opts, "cal_half_window", 20);
    const int n_s_extend = opt_int(opts, "n_s_extend", 4);
    const int n_cores = opt_int(opts, "n_cores", 4);
    const int n_group = opt_int(opts, "n_group", 4);
    const double energy = opt_double(opts, "energy", 14000.0);
    const double p_x = opt_double(opts, "p_x", 0.65e-6);
    const double z = opt_double(opts, "z", 0.5);
    const int wavelet_level_cut = opt_int(opts, "wavelet_level_cut", 2);
    const int pyramid_level = opt_int(opts, "pyramid_level", 2);
    const int n_iter = opt_int(opts, "n_iter", 1);
    const bool use_estimate = opt_bool(opts, "use_estimate", false);
    const bool use_wavelet = opt_bool(opts, "use_wavelet", true);
    const int use_gpu = opt_bool(opts, "use_gpu", false) ? 1 : 0;
    const int wavelet_impl = opt_int(opts, "wavelet_impl", 0);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WXST wxst(
        img.data, ref.data, img.h, img.w,
        m_image, n_s, cal_half_window, n_s_extend, n_cores, n_group,
        energy, p_x, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, wavelet_impl);
    const auto out = wxst.run(out_dir, h5_deflate);
    const auto t_save_t0 = std::chrono::steady_clock::now();
    if (save_img) {
        const std::filesystem::path od(out_dir);
        wsvt::save_img(out.displace_x, out.h, out.w, (od / "displace_x.tif").string());
        wsvt::save_img(out.displace_y, out.h, out.w, (od / "displace_y.tif").string());
        wsvt::save_img(out.dpc_x, out.h, out.w, (od / "DPC_x.tif").string());
        wsvt::save_img(out.dpc_y, out.h, out.w, (od / "DPC_y.tif").string());
        wsvt::save_img(out.phase, out.h, out.w, (od / "phase.tif").string());
        wsvt::save_img(out.transmission, out.transmission_h, out.transmission_w, (od / "transmission_image.tif").string());
        wsvt::save_img(out.darkfield_nd, out.h, out.w, (od / "darkfield_nd.tif").string());
    }
    const auto t_save_t1 = std::chrono::steady_clock::now();
    const double save_time_s = std::chrono::duration<double>(t_save_t1 - t_save_t0).count();
    const auto t_process_t1 = std::chrono::steady_clock::now();
    const double process_wall_s = std::chrono::duration<double>(t_process_t1 - t_process_t0).count();
    std::cout << "=== Timing Summary ===" << std::endl;
    std::cout << "  load:             " << load_time_s << " s" << std::endl;
    std::cout << "  pyramid:          " << out.pyramid_time_s << " s" << std::endl;
    std::cout << "  template_window:  " << out.template_window_time_s << " s" << std::endl;
    std::cout << "  wavelet:          " << out.wavelet_time_s << " s" << std::endl;
    std::cout << "  displace:         " << out.displace_time_s << " s" << std::endl;
    std::cout << "  post-proc:        " << out.postprocess_time_s << " s" << std::endl;
    std::cout << "  result write:     " << out.result_write_time_s << " s"
              << " (h5_deflate=" << h5_deflate << ")" << std::endl;
    std::cout << "  optional_image_save: " << save_time_s << " s" << std::endl;
    std::cout << "  solver:           " << out.time_cost_s << " s" << std::endl;
    std::cout << "  wall total:       " << (load_time_s + out.time_cost_s + out.result_write_time_s + save_time_s) << " s" << std::endl;
    std::cout << "  process wall:     " << process_wall_s << " s" << std::endl;
    std::cout << "wxst_dir done: " << out.h << "x" << out.w << std::endl;
    return 0;
}

int run_wsvt_cmd(const std::string& img_h5, const std::string& img_key, const std::string& ref_h5, const std::string& ref_key, const std::string& out_dir, const OptMap& opts) {
    const auto t_process_t0 = std::chrono::steady_clock::now();
    const auto t_load_t0 = std::chrono::steady_clock::now();
    const auto img = wsvt::read_h5(img_h5, img_key, false);
    const auto ref = wsvt::read_h5(ref_h5, ref_key, false);
    if (img.shape.size() != 3 || ref.shape.size() != 3) {
        throw std::invalid_argument("wsvt input must be 3D [ch,h,w]");
    }
    if (img.shape != ref.shape) {
        throw std::invalid_argument("wsvt img/ref shape mismatch");
    }
    const auto t_load_t1 = std::chrono::steady_clock::now();
    const double load_time_s = std::chrono::duration<double>(t_load_t1 - t_load_t0).count();
    const std::size_t ch = img.shape[0];
    const std::size_t h = img.shape[1];
    const std::size_t w = img.shape[2];
    const int crop = opt_int(opts, "crop", 512);
    const int cal_half_window = opt_int(opts, "cal_half_window", 20);
    const int n_template = opt_int(opts, "n_template", 0);
    const int n_s_extend = opt_int(opts, "n_s_extend", 4);
    const int n_cores = opt_int(opts, "n_cores", 4);
    const int n_group = opt_int(opts, "n_group", 4);
    const double energy = opt_double(opts, "energy", 14000.0);
    const double p_x = opt_double(opts, "p_x", 0.65e-6);
    const double mag_factor = opt_double(opts, "mag_factor", 1.0);
    const double z = opt_double(opts, "z", 0.5);
    const int wavelet_level_cut = opt_int(opts, "wavelet_level_cut", 2);
    const int pyramid_level = opt_int(opts, "pyramid_level", 2);
    const int n_iter = opt_int(opts, "n_iter", 1);
    const bool use_estimate = opt_bool(opts, "use_estimate", false);
    const bool use_wavelet = opt_bool(opts, "use_wavelet", true);
    const int use_gpu = opt_bool(opts, "use_gpu", false) ? 1 : 0;
    const bool calc_darkfield = opt_bool(opts, "calc_darkfield", true);
    const bool cleansave = opt_bool(opts, "cleansave", false);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WSVT wsvt_solver(
        img.data, ref.data, ch, h, w,
        crop, cal_half_window, n_template, n_s_extend, n_cores, n_group,
        energy, p_x, mag_factor, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, calc_darkfield);
    const auto out = wsvt_solver.run(out_dir, cleansave, h5_deflate);
    const auto t_save_t0 = std::chrono::steady_clock::now();
    if (save_img) {
        const std::filesystem::path od(out_dir);
        wsvt::save_img(out.displace_x, out.h, out.w, (od / "displace_x.tif").string());
        wsvt::save_img(out.displace_y, out.h, out.w, (od / "displace_y.tif").string());
        wsvt::save_img(out.dpc_x, out.h, out.w, (od / "DPC_x.tif").string());
        wsvt::save_img(out.dpc_y, out.h, out.w, (od / "DPC_y.tif").string());
        wsvt::save_img(out.phase, out.h, out.w, (od / "phase.tif").string());
        wsvt::save_img(out.darkfield_nd, out.h, out.w, (od / "darkfield_nd.tif").string());
        wsvt::save_img(out.transmission, out.transmission_h, out.transmission_w, (od / "transmission_image.tif").string());
        if (!out.darkfield.empty()) {
            wsvt::save_img(out.darkfield, out.transmission_h, out.transmission_w, (od / "darkfield.tif").string());
        }
    }
    const auto t_save_t1 = std::chrono::steady_clock::now();
    const double save_time_s = std::chrono::duration<double>(t_save_t1 - t_save_t0).count();
    const auto t_process_t1 = std::chrono::steady_clock::now();
    const double process_wall_s = std::chrono::duration<double>(t_process_t1 - t_process_t0).count();
    std::cout << "=== Timing Summary ===" << std::endl;
    std::cout << "  load:             " << load_time_s << " s" << std::endl;
    std::cout << "  pyramid:          " << out.pyramid_time_s << " s" << std::endl;
    std::cout << "  template_window:  " << out.template_window_time_s << " s" << std::endl;
    std::cout << "  wavelet:          " << out.wavelet_time_s << " s" << std::endl;
    std::cout << "  raw darkfield:    " << out.darkfield_time_s << " s"
              << (calc_darkfield ? "" : " (disabled)") << std::endl;
    std::cout << "  displace:         " << out.displace_time_s << " s" << std::endl;
    std::cout << "  post-proc:        " << out.postprocess_time_s << " s" << std::endl;
    std::cout << "  result write:     " << out.result_write_time_s << " s"
              << " (h5_deflate=" << h5_deflate << ")" << std::endl;
    std::cout << "  optional_image_save: " << save_time_s << " s" << std::endl;
    std::cout << "  solver:           " << out.time_cost_s << " s" << std::endl;
    std::cout << "  wall total:       " << (load_time_s + out.time_cost_s + out.result_write_time_s + save_time_s) << " s" << std::endl;
    std::cout << "  process wall:     " << process_wall_s << " s" << std::endl;
    std::cout << "wsvt done: " << out.h << "x" << out.w << std::endl;
    return 0;
}

int run_wsvt_dir_cmd(const std::string& img_dir, const std::string& ref_dir, const std::string& out_dir, const OptMap& opts) {
    const auto t_process_t0 = std::chrono::steady_clock::now();
    const auto t_load_t0 = std::chrono::steady_clock::now();
    const auto img_files = wsvt::list_image_files(img_dir);
    const auto ref_files = wsvt::list_image_files(ref_dir);
    if (img_files.size() != ref_files.size()) {
        throw std::invalid_argument("wsvt_dir expects same image count in img/ref folders");
    }
    if (img_files.empty()) {
        throw std::invalid_argument("wsvt_dir no input files");
    }
    // Read first frame to get dimensions
    const auto first_img = wsvt::read_image_gray(img_files[0]);
    const std::size_t h = first_img.h;
    const std::size_t w = first_img.w;
    const std::size_t ch = img_files.size();
    std::vector<float> img_stack(ch * h * w, 0.0f);
    std::vector<float> ref_stack(ch * h * w, 0.0f);
    std::memcpy(img_stack.data(), first_img.data.data(), h * w * sizeof(float));
    wsvt::read_image_gray_into(ref_files[0], ref_stack.data(), h, w);

    // Parallel read remaining frames: decode directly into pre-allocated stack.
    // Cap I/O threads to avoid saturating disk/decoder with too many concurrent reads.
    if (ch > 1) {
        const int io_threads = std::min(static_cast<int>(ch) - 1, 32);
        #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads)
        for (std::size_t i = 1; i < ch; ++i) {
            wsvt::read_image_gray_into(img_files[i], img_stack.data() + i * h * w, h, w);
            wsvt::read_image_gray_into(ref_files[i], ref_stack.data() + i * h * w, h, w);
        }
    }
    const auto t_load_t1 = std::chrono::steady_clock::now();
    const double load_time_s = std::chrono::duration<double>(t_load_t1 - t_load_t0).count();
    std::cout << "load time: " << load_time_s << " s" << std::endl;

    const int crop = opt_int(opts, "crop", 512);

    // Crop before alignment to reduce FFT cost
    std::size_t work_h = h;
    std::size_t work_w = w;
    if (crop > 0 && static_cast<std::size_t>(crop) < std::min(h, w)) {
        const std::size_t crop_sz = static_cast<std::size_t>(crop);
        const std::size_t y0 = (h - crop_sz) / 2;
        const std::size_t x0 = (w - crop_sz) / 2;
        std::vector<float> img_cropped(ch * crop_sz * crop_sz);
        std::vector<float> ref_cropped(ch * crop_sz * crop_sz);
        for (std::size_t i = 0; i < ch; ++i) {
            for (std::size_t y = 0; y < crop_sz; ++y) {
                std::memcpy(img_cropped.data() + i * crop_sz * crop_sz + y * crop_sz,
                            img_stack.data() + i * h * w + (y0 + y) * w + x0,
                            crop_sz * sizeof(float));
                std::memcpy(ref_cropped.data() + i * crop_sz * crop_sz + y * crop_sz,
                            ref_stack.data() + i * h * w + (y0 + y) * w + x0,
                            crop_sz * sizeof(float));
            }
        }
        img_stack = std::move(img_cropped);
        ref_stack = std::move(ref_cropped);
        work_h = crop_sz;
        work_w = crop_sz;
    }

    const bool align = opt_bool(opts, "align", false);
    double align_time_s = 0.0;
    if (align) {
        const auto t_align_t0 = std::chrono::steady_clock::now();
        const auto align_result = wsvt::stack_image_align(
            ref_stack,
            img_stack.data(), work_w,
            ch, work_h, work_w);
        (void)align_result;
        const auto t_align_t1 = std::chrono::steady_clock::now();
        align_time_s = std::chrono::duration<double>(t_align_t1 - t_align_t0).count();
    }
    std::cout << "align time: " << align_time_s << " s" << (align ? "" : " (disabled)") << std::endl;
    const int cal_half_window = opt_int(opts, "cal_half_window", 20);
    const int n_template = opt_int(opts, "n_template", 0);
    const int n_s_extend = opt_int(opts, "n_s_extend", 4);
    const int n_cores = opt_int(opts, "n_cores", 4);
    const int n_group = opt_int(opts, "n_group", 4);
    const double energy = opt_double(opts, "energy", 14000.0);
    const double p_x = opt_double(opts, "p_x", 0.65e-6);
    const double mag_factor = opt_double(opts, "mag_factor", 1.0);
    const double z = opt_double(opts, "z", 0.5);
    const int wavelet_level_cut = opt_int(opts, "wavelet_level_cut", 2);
    const int pyramid_level = opt_int(opts, "pyramid_level", 2);
    const int n_iter = opt_int(opts, "n_iter", 1);
    const bool use_estimate = opt_bool(opts, "use_estimate", false);
    const bool use_wavelet = opt_bool(opts, "use_wavelet", true);
    const int use_gpu = opt_bool(opts, "use_gpu", false) ? 1 : 0;
    const bool calc_darkfield = opt_bool(opts, "calc_darkfield", true);
    const bool cleansave = opt_bool(opts, "cleansave", false);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WSVT wsvt_solver(
        img_stack, ref_stack, ch, work_h, work_w,
        0, cal_half_window, n_template, n_s_extend, n_cores, n_group,
        energy, p_x, mag_factor, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, calc_darkfield);
    const auto out = wsvt_solver.run(out_dir, cleansave, h5_deflate);
    const auto t_save_t0 = std::chrono::steady_clock::now();
    if (save_img) {
        const std::filesystem::path od(out_dir);
        wsvt::save_img(out.displace_x, out.h, out.w, (od / "displace_x.tif").string());
        wsvt::save_img(out.displace_y, out.h, out.w, (od / "displace_y.tif").string());
        wsvt::save_img(out.dpc_x, out.h, out.w, (od / "DPC_x.tif").string());
        wsvt::save_img(out.dpc_y, out.h, out.w, (od / "DPC_y.tif").string());
        wsvt::save_img(out.phase, out.h, out.w, (od / "phase.tif").string());
        wsvt::save_img(out.darkfield_nd, out.h, out.w, (od / "darkfield_nd.tif").string());
        wsvt::save_img(out.transmission, out.transmission_h, out.transmission_w, (od / "transmission_image.tif").string());
        if (!out.darkfield.empty()) {
            wsvt::save_img(out.darkfield, out.transmission_h, out.transmission_w, (od / "darkfield.tif").string());
        }
    }
    const auto t_save_t1 = std::chrono::steady_clock::now();
    const double save_time_s = std::chrono::duration<double>(t_save_t1 - t_save_t0).count();
    const auto t_process_t1 = std::chrono::steady_clock::now();
    const double process_wall_s = std::chrono::duration<double>(t_process_t1 - t_process_t0).count();
    std::cout << "=== Timing Summary ===" << std::endl;
    std::cout << "  load:             " << load_time_s << " s" << std::endl;
    std::cout << "  align:            " << align_time_s << " s" << std::endl;
    std::cout << "  pyramid:          " << out.pyramid_time_s << " s" << std::endl;
    std::cout << "  template_window:  " << out.template_window_time_s << " s" << std::endl;
    std::cout << "  wavelet:          " << out.wavelet_time_s << " s" << std::endl;
    std::cout << "  raw darkfield:    " << out.darkfield_time_s << " s"
              << (calc_darkfield ? "" : " (disabled)") << std::endl;
    std::cout << "  displace:         " << out.displace_time_s << " s" << std::endl;
    std::cout << "  post-proc:        " << out.postprocess_time_s << " s" << std::endl;
    std::cout << "  result write:     " << out.result_write_time_s << " s"
              << " (h5_deflate=" << h5_deflate << ")" << std::endl;
    std::cout << "  optional_image_save: " << save_time_s << " s" << std::endl;
    std::cout << "  solver:           " << out.time_cost_s << " s" << std::endl;
    std::cout << "  wall total:       " << (load_time_s + align_time_s + out.time_cost_s + out.result_write_time_s + save_time_s) << " s" << std::endl;
    std::cout << "  process wall:     " << process_wall_s << " s" << std::endl;
    std::cout << "wsvt_dir done: " << out.h << "x" << out.w << std::endl;
    return 0;
}

void print_usage() {
    std::cout << "usage:\n"
              << "  wsvt_cli demo\n"
              << "  wsvt_cli wxst <img_h5> <img_key> <ref_h5> <ref_key> <out_dir> [--key value ...]\n"
              << "  wsvt_cli wsvt <img_h5> <img_key> <ref_h5> <ref_key> <out_dir> [--key value ...]\n"
              << "  wsvt_cli wxst_dir <img_dir> <ref_dir> <out_dir> [--key value ...]\n"
              << "  wsvt_cli wsvt_dir <img_dir> <ref_dir> <out_dir> [--key value ...]\n";
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            return run_demo();
        }
        const std::string cmd = argv[1];
        if (cmd == "demo") {
            return run_demo();
        }
        if (cmd == "wxst") {
            if (argc < 7) {
                print_usage();
                return 2;
            }
            const auto opts = parse_opts(argc, argv, 7);
            return run_wxst_cmd(argv[2], argv[3], argv[4], argv[5], argv[6], opts);
        }
        if (cmd == "wsvt") {
            if (argc < 7) {
                print_usage();
                return 2;
            }
            const auto opts = parse_opts(argc, argv, 7);
            return run_wsvt_cmd(argv[2], argv[3], argv[4], argv[5], argv[6], opts);
        }
        if (cmd == "wxst_dir") {
            if (argc < 5) {
                print_usage();
                return 2;
            }
            const auto opts = parse_opts(argc, argv, 5);
            return run_wxst_dir_cmd(argv[2], argv[3], argv[4], opts);
        }
        if (cmd == "wsvt_dir") {
            if (argc < 5) {
                print_usage();
                return 2;
            }
            const auto opts = parse_opts(argc, argv, 5);
            return run_wsvt_dir_cmd(argv[2], argv[3], argv[4], opts);
        }
        print_usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
