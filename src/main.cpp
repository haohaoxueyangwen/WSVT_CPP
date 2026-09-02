#include "wsvt/io_h5.hpp"
#include "wsvt/io_image.hpp"
#include "wsvt/io_json.hpp"
#include "wsvt/wsvt_pipeline.hpp"
#include "wsvt/wxst_pipeline.hpp"

#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

std::string opt_string(
    const OptMap& opts, const std::string& key, const std::string& def = "") {
    const auto it = opts.find(key);
    return it == opts.end() ? def : it->second;
}

std::vector<std::size_t> opt_frame_stages(
    const OptMap& opts,
    std::size_t frame_count) {
    const std::string text = opt_string(opts, "frame_stages");
    std::vector<std::size_t> stages;
    if (!text.empty()) {
        std::istringstream input(text);
        std::string item;
        while (std::getline(input, item, ',')) {
            if (item.empty()) {
                throw std::invalid_argument("frame_stages contains an empty item");
            }
            stages.push_back(static_cast<std::size_t>(std::stoull(item)));
        }
        return stages;
    }
    for (const std::size_t value : {std::size_t{8}, std::size_t{16},
                                    std::size_t{25}, std::size_t{50},
                                    std::size_t{100}}) {
        if (value < frame_count) {
            stages.push_back(value);
        }
    }
    if (stages.empty() && frame_count >= 4) {
        stages.push_back(frame_count / 2);
    }
    stages.push_back(frame_count);
    return stages;
}

void configure_easy_to_hard_from_options(
    const OptMap& opts,
    std::size_t frame_count,
    wsvt::WSVT& solver) {
    const bool adaptive_frames = opt_bool(opts, "adaptive_frames", false);
    const bool confidence =
        opt_bool(opts, "confidence_aware_refinement", false) || adaptive_frames;
    if (!confidence) {
        return;
    }
    wsvt::EasyToHardConfig config;
    config.confidence_aware_refinement = true;
    config.adaptive_frames = adaptive_frames;
    config.lazy_temporal_descriptors =
        opt_bool(opts, "lazy_temporal_descriptors", false);
    config.prefix_compatible_temporal_wavelet =
        opt_bool(opts, "prefix_compatible_temporal_wavelet", false);
    config.temporal_probe_first_stage_only =
        opt_bool(opts, "temporal_probe_first_stage_only", false);
    config.easy_half_window = opt_int(opts, "easy_half_window", 1);
    config.temporal_prefix_half_window =
        opt_int(opts, "temporal_prefix_half_window", 1);
    config.score_margin_min = static_cast<float>(
        opt_double(opts, "confidence_margin_min", 0.02));
    config.normalized_curvature_min = static_cast<float>(
        opt_double(opts, "confidence_curvature_min", 0.0));
    config.temporal_speckle_contrast_min = static_cast<float>(
        opt_double(opts, "confidence_contrast_min", 0.0));
    config.interlevel_delta_max_px = static_cast<float>(
        opt_double(opts, "confidence_interlevel_delta_max", 1.5));
    config.temporal_delta_max_px = static_cast<float>(
        opt_double(opts, "confidence_temporal_delta_max", 0.5));
    if (adaptive_frames) {
        config.frame_stages = opt_frame_stages(opts, frame_count);
    }
    solver.configure_easy_to_hard(std::move(config));
}

void configure_wavelet_guided_umpa_from_options(
    const OptMap& opts,
    wsvt::WSVT& solver) {
    if (!opt_bool(opts, "wavelet_guided_umpa", false)) {
        return;
    }
    wsvt::WaveletGuidedUmpaConfig config;
    config.enabled = true;
    config.local_half_window = opt_int(opts, "umpa_local_half_window", 1);
    config.analysis_radius = static_cast<std::size_t>(
        opt_int(opts, "umpa_analysis_radius", 1));
    config.relative_delta_tolerance = opt_double(
        opts, "umpa_relative_delta_tolerance", 1.0e-12);
    config.transmission_epsilon = opt_double(
        opts, "umpa_transmission_epsilon", 1.0e-12);
    solver.configure_wavelet_guided_umpa(std::move(config));
}

void configure_fixed_set_transport_from_options(
    const OptMap& opts,
    wsvt::WSVT& solver) {
    if (opt_bool(opts, "fixed_set_transport", false)) {
        solver.configure_fixed_set_transport(true);
    }
}

wsvt::SetTransportRawObjective parse_set_transport_raw_objective(
    std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (value == "windowed_zncc" || value == "zncc") {
        return wsvt::SetTransportRawObjective::WindowedZncc;
    }
    if (value == "modelt" || value == "model_t") {
        return wsvt::SetTransportRawObjective::ModelT;
    }
    if (value == "modeldf" || value == "model_df") {
        return wsvt::SetTransportRawObjective::ModelDF;
    }
    throw std::invalid_argument(
        "raw_rerank_objective must be windowed_zncc, ModelT, or ModelDF");
}

void configure_set_transport_raw_rerank_from_options(
    const OptMap& opts,
    wsvt::WSVT& solver) {
    if (!opt_bool(opts, "set_transport_raw_rerank", false)) {
        return;
    }
    wsvt::SetTransportRawRerankConfig config;
    config.enabled = true;
    config.representatives_only = opt_bool(
        opts, "set_transport_raw_representatives", false);
    config.objective = parse_set_transport_raw_objective(
        opt_string(opts, "raw_rerank_objective", "ModelDF"));
    config.domain_half_window = 4;
    config.analysis_radius = 1;
    config.relative_delta_tolerance = opt_double(
        opts, "umpa_relative_delta_tolerance", 1.0e-12);
    config.transmission_epsilon = opt_double(
        opts, "umpa_transmission_epsilon", 1.0e-12);
    config.variance_epsilon = opt_double(
        opts, "raw_rerank_variance_epsilon", 1.0e-12);
    solver.configure_set_transport_raw_rerank(std::move(config));
}

struct DiagnosticPointSpec {
    std::string point_id;
    std::string roi_id;
    std::size_t raw_y = 0;
    std::size_t raw_x = 0;
};

std::vector<DiagnosticPointSpec> read_diagnostic_points(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open diagnostic point file: " + path);
    }
    std::vector<DiagnosticPointSpec> points;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::istringstream row(line);
        DiagnosticPointSpec point;
        if (!(row >> point.point_id >> point.roi_id >> point.raw_y >> point.raw_x)) {
            throw std::invalid_argument(
                "diagnostic point rows must be: point_id roi_id raw_y raw_x");
        }
        std::string trailing;
        if (row >> trailing) {
            throw std::invalid_argument("unexpected column in diagnostic point row");
        }
        points.push_back(std::move(point));
    }
    if (points.empty()) {
        throw std::invalid_argument("diagnostic point file is empty");
    }
    return points;
}

void write_topk_diagnostics(
    const std::filesystem::path& path,
    const std::vector<DiagnosticPointSpec>& points,
    const std::vector<wsvt::SearchTopKDiagnostic>& diagnostics,
    std::size_t width) {
    std::unordered_map<std::size_t, const DiagnosticPointSpec*> point_by_pixel;
    for (const auto& point : points) {
        const std::size_t pixel = point.raw_y * width + point.raw_x;
        if (!point_by_pixel.emplace(pixel, &point).second) {
            throw std::invalid_argument("duplicate diagnostic raw pixel");
        }
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write Top-K diagnostic CSV: " + path.string());
    }
    output << "point_id,roi_id,pyramid_level,raw_y,raw_x,rank,initial_guess_y,initial_guess_x,"
              "local_offset_y,local_offset_x,internal_candidate_y,internal_candidate_x,"
              "saved_candidate_y,saved_candidate_x,descriptor_score_neg_ssd,"
              "descriptor_cost_ssd\n";
    output << std::setprecision(9);
    for (const auto& diagnostic : diagnostics) {
        const std::size_t pixel = diagnostic.raw_y * width + diagnostic.raw_x;
        const auto found = point_by_pixel.find(pixel);
        if (found == point_by_pixel.end()) {
            throw std::logic_error("solver returned an unrequested diagnostic pixel");
        }
        const auto& point = *found->second;
        output << point.point_id << ',' << point.roi_id << ','
               << diagnostic.pyramid_level << ',' << diagnostic.raw_y << ','
               << diagnostic.raw_x << ','
               << diagnostic.rank << ',' << diagnostic.initial_guess_y << ','
               << diagnostic.initial_guess_x << ',' << diagnostic.local_offset_y << ','
               << diagnostic.local_offset_x << ',' << diagnostic.internal_candidate_y << ','
               << diagnostic.internal_candidate_x << ',' << diagnostic.saved_candidate_y << ','
               << diagnostic.saved_candidate_x << ','
               << diagnostic.descriptor_score_neg_ssd << ','
               << diagnostic.descriptor_cost_ssd << '\n';
    }
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
    const int phase_cores = opt_int(opts, "phase_cores", n_cores);
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
    const int wavelet_impl = opt_int(opts, "wavelet_impl", 2);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WXST wxst(
        img.data, ref.data, h, w,
        m_image, n_s, cal_half_window, n_s_extend, n_cores, n_group,
        energy, p_x, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, wavelet_impl, phase_cores);
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
    const int phase_cores = opt_int(opts, "phase_cores", n_cores);
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
    const int wavelet_impl = opt_int(opts, "wavelet_impl", 2);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WXST wxst(
        img.data, ref.data, img.h, img.w,
        m_image, n_s, cal_half_window, n_s_extend, n_cores, n_group,
        energy, p_x, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, wavelet_impl, phase_cores);
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
    auto img = wsvt::read_h5(img_h5, img_key, false);
    auto ref = wsvt::read_h5(ref_h5, ref_key, false);
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
    const int phase_cores = opt_int(opts, "phase_cores", n_cores);
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
    const bool search_early_abandon = opt_bool(opts, "search_early_abandon", false);
    const bool search_two_pass = opt_bool(opts, "search_two_pass", false);
    const bool search_guard_cache = opt_bool(opts, "search_guard_cache", false);
    const int search_top_k = opt_int(opts, "search_top_k", 2);
    const int search_block_size = opt_int(opts, "search_block_size", 16);
    const int search_prefix_size = opt_int(opts, "search_prefix_size", 16);
    const bool cleansave = opt_bool(opts, "cleansave", false);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WSVT wsvt_solver(
        std::move(img.data), std::move(ref.data), ch, h, w,
        crop, cal_half_window, n_template, n_s_extend, n_cores, n_group,
        energy, p_x, mag_factor, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, calc_darkfield, phase_cores,
        search_early_abandon, search_top_k, search_block_size,
        search_two_pass, search_prefix_size, search_guard_cache);
    configure_easy_to_hard_from_options(opts, ch, wsvt_solver);
    configure_wavelet_guided_umpa_from_options(opts, wsvt_solver);
    configure_fixed_set_transport_from_options(opts, wsvt_solver);
    configure_set_transport_raw_rerank_from_options(opts, wsvt_solver);
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
    std::cout << "  search pruning:   " << out.search_abandoned_candidate_count
              << "/" << out.search_candidate_count << " candidates, "
              << out.search_distance_terms_evaluated << "/"
              << out.search_distance_terms_possible << " main terms, "
              << out.search_refine_terms_evaluated << " refine terms, "
              << out.search_prefix_terms_evaluated << " prefix terms, "
              << out.search_full_candidate_count << " complete candidates" << std::endl;
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

int run_wsvt_stage_cmd(
    const std::string& img_h5,
    const std::string& img_key,
    const std::string& ref_h5,
    const std::string& ref_key,
    const std::string& out_dir,
    const OptMap& opts) {
    auto img = wsvt::read_h5(img_h5, img_key, false);
    auto ref = wsvt::read_h5(ref_h5, ref_key, false);
    if (img.shape.size() != 3 || ref.shape.size() != 3 || img.shape != ref.shape) {
        throw std::invalid_argument("wsvt_stage expects matching 3D [ch,h,w] inputs");
    }

    const std::size_t ch = img.shape[0];
    const std::size_t h = img.shape[1];
    const std::size_t w = img.shape[2];
    const int crop = opt_int(opts, "crop", 0);
    const int cal_half_window = opt_int(opts, "cal_half_window", 2);
    const int n_template = opt_int(opts, "n_template", 0);
    const int n_s_extend = opt_int(opts, "n_s_extend", 1);
    const int n_cores = opt_int(opts, "n_cores", 1);
    const int phase_cores = opt_int(opts, "phase_cores", n_cores);
    const double energy = opt_double(opts, "energy", 14000.0);
    const double p_x = opt_double(opts, "p_x", 0.65e-6);
    const double mag_factor = opt_double(opts, "mag_factor", 1.0);
    const double z = opt_double(opts, "z", 0.5);
    const int wavelet_level_cut = opt_int(opts, "wavelet_level_cut", 1);
    const int pyramid_level = opt_int(opts, "pyramid_level", 1);
    const int h5_deflate = opt_int(opts, "h5_deflate", 0);

    const auto make_solver = [&](std::vector<float> img_data,
                                 std::vector<float> ref_data,
                                 bool use_wavelet) {
        return wsvt::WSVT(
            std::move(img_data), std::move(ref_data), ch, h, w,
            crop, cal_half_window, n_template, n_s_extend, n_cores,
            /*n_group=*/1, energy, p_x, mag_factor, z, wavelet_level_cut,
            pyramid_level, /*n_iter=*/1, /*use_estimate=*/false,
            use_wavelet, /*use_gpu=*/0, /*calc_darkfield=*/false, phase_cores);
    };

    auto prewavelet_solver = make_solver(img.data, ref.data, false);
    auto prewavelet = prewavelet_solver.pyramid_data();
    auto descriptor_solver = make_solver(std::move(img.data), std::move(ref.data), true);
    auto descriptor = descriptor_solver.wavelet_data();
    auto final_solver = make_solver(
        wsvt::read_h5(img_h5, img_key, false).data,
        wsvt::read_h5(ref_h5, ref_key, false).data,
        true);
    auto final_output = final_solver.solver();

    std::vector<wsvt::H5ItemF32> items;
    const auto append_levels = [&](const char* stage,
                                   const std::vector<wsvt::PyramidLevel>& img_levels,
                                   const std::vector<wsvt::PyramidLevel>& ref_levels) {
        for (std::size_t level = 0; level < img_levels.size(); ++level) {
            const auto append_one = [&](const char* role, const wsvt::PyramidLevel& value) {
                const std::string key = std::string(stage) + "_" + role +
                                        "_hwd_l" + std::to_string(level);
                std::vector<float> data(value.data.begin(), value.data.end());
                items.push_back(wsvt::H5ItemF32{
                    key,
                    wsvt::NdArrayF32{{value.d1, value.d2, value.d0}, std::move(data)}});
            };
            append_one("img", img_levels[level]);
            append_one("ref", ref_levels[level]);
        }
    };
    append_levels("prewavelet", prewavelet.img_levels, prewavelet.ref_levels);
    append_levels("descriptor", descriptor.img_levels, descriptor.ref_levels);
    const auto append_image = [&](const char* key,
                                  std::size_t image_h,
                                  std::size_t image_w,
                                  const std::vector<float>& values) {
        items.push_back(wsvt::H5ItemF32{
            key, wsvt::NdArrayF32{{image_h, image_w}, values}});
    };
    append_image("displace_y_hw", final_output.h, final_output.w, final_output.displace_y);
    append_image("displace_x_hw", final_output.h, final_output.w, final_output.displace_x);
    append_image("dpc_y_hw", final_output.h, final_output.w, final_output.dpc_y);
    append_image("dpc_x_hw", final_output.h, final_output.w, final_output.dpc_x);
    append_image("phase_hw", final_output.h, final_output.w, final_output.phase);
    append_image(
        "transmission_hw", final_output.transmission_h, final_output.transmission_w,
        final_output.transmission);

    std::filesystem::create_directories(out_dir);
    wsvt::write_h5(out_dir, "WSVT_stage", items, h5_deflate);
    wsvt::JsonObject manifest;
    manifest["debug_only"] = true;
    manifest["semantics_profile"] = "python_reference_v1";
    manifest["window_policy"] = "manual_fixed";
    manifest["crop"] = static_cast<double>(crop);
    manifest["cal_half_window"] = static_cast<double>(cal_half_window);
    manifest["n_template"] = static_cast<double>(n_template);
    manifest["n_s_extend"] = static_cast<double>(n_s_extend);
    manifest["pyramid_level"] = static_cast<double>(pyramid_level);
    manifest["wavelet_level_cut"] = static_cast<double>(wavelet_level_cut);
    manifest["dtype"] = "float32";
    manifest["axis_order"] = "HWD";
    wsvt::write_json(out_dir, "WSVT_stage", manifest);
    std::cout << "wsvt_stage done: " << items.size() << " arrays" << std::endl;
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
    const int phase_cores = opt_int(opts, "phase_cores", n_cores);
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
    const bool search_early_abandon = opt_bool(opts, "search_early_abandon", false);
    const bool search_two_pass = opt_bool(opts, "search_two_pass", false);
    const bool search_guard_cache = opt_bool(opts, "search_guard_cache", false);
    const int search_top_k = opt_int(opts, "search_top_k", 2);
    const int search_block_size = opt_int(opts, "search_block_size", 16);
    const int search_prefix_size = opt_int(opts, "search_prefix_size", 16);
    const bool cleansave = opt_bool(opts, "cleansave", false);
    const bool save_img = opt_bool(opts, "save_img", false);
    const int h5_deflate = opt_int(opts, "h5_deflate", 9);
    std::filesystem::create_directories(out_dir);
    wsvt::WSVT wsvt_solver(
        std::move(img_stack), std::move(ref_stack), ch, work_h, work_w,
        0, cal_half_window, n_template, n_s_extend, n_cores, n_group,
        energy, p_x, mag_factor, z, wavelet_level_cut, pyramid_level, n_iter,
        use_estimate, use_wavelet, use_gpu, calc_darkfield, phase_cores,
        search_early_abandon, search_top_k, search_block_size,
        search_two_pass, search_prefix_size, search_guard_cache);
    configure_easy_to_hard_from_options(opts, ch, wsvt_solver);
    configure_wavelet_guided_umpa_from_options(opts, wsvt_solver);
    configure_fixed_set_transport_from_options(opts, wsvt_solver);
    configure_set_transport_raw_rerank_from_options(opts, wsvt_solver);
    const std::string diagnostic_points_path = opt_string(opts, "diagnostic_points");
    std::vector<DiagnosticPointSpec> diagnostic_points;
    std::size_t diagnostic_grid_w = work_w;
    if (!diagnostic_points_path.empty()) {
        if (crop > 0 && static_cast<std::size_t>(crop) < std::min(h, w)) {
            throw std::invalid_argument("Top-K diagnostic points require --crop 0");
        }
        diagnostic_points = read_diagnostic_points(diagnostic_points_path);
        const int diagnostic_pyramid_level = opt_int(
            opts, "diagnostic_pyramid_level", 0);
        if (diagnostic_pyramid_level < 0 || diagnostic_pyramid_level > pyramid_level) {
            throw std::invalid_argument(
                "diagnostic_pyramid_level must be within the configured pyramid");
        }
        std::size_t diagnostic_grid_h = work_h;
        for (int level = 0; level < diagnostic_pyramid_level; ++level) {
            diagnostic_grid_h = (diagnostic_grid_h + 5U) / 2U;
            diagnostic_grid_w = (diagnostic_grid_w + 5U) / 2U;
        }
        std::vector<std::size_t> pixels;
        pixels.reserve(diagnostic_points.size());
        for (const auto& point : diagnostic_points) {
            if (point.raw_y >= diagnostic_grid_h || point.raw_x >= diagnostic_grid_w) {
                throw std::out_of_range(
                    "diagnostic point is outside the selected pyramid grid");
            }
            pixels.push_back(point.raw_y * diagnostic_grid_w + point.raw_x);
        }
        wsvt_solver.configure_search_topk_diagnostics(
            std::move(pixels),
            static_cast<std::size_t>(opt_int(opts, "diagnostic_top_k", 4)),
            diagnostic_pyramid_level);
    }
    const auto out = diagnostic_points.empty()
        ? wsvt_solver.run(out_dir, cleansave, h5_deflate)
        : wsvt_solver.solver();
    if (!diagnostic_points.empty()) {
        std::filesystem::create_directories(out_dir);
        const auto diagnostic_csv = std::filesystem::path(out_dir) /
            opt_string(opts, "diagnostic_output_name", "topk_descriptor_candidates.csv");
        write_topk_diagnostics(
            diagnostic_csv,
            diagnostic_points,
            wsvt_solver.search_topk_diagnostics(),
            diagnostic_grid_w);
        std::cout << "diagnostic Top-K CSV: " << diagnostic_csv << std::endl;
    }
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
    std::cout << "  search pruning:   " << out.search_abandoned_candidate_count
              << "/" << out.search_candidate_count << " candidates, "
              << out.search_distance_terms_evaluated << "/"
              << out.search_distance_terms_possible << " main terms, "
              << out.search_refine_terms_evaluated << " refine terms, "
              << out.search_prefix_terms_evaluated << " prefix terms, "
              << out.search_full_candidate_count << " complete candidates" << std::endl;
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
              << "  wsvt_cli wsvt_stage <img_h5> <img_key> <ref_h5> <ref_key> <out_dir> [--key value ...]\n"
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
        if (cmd == "wsvt_stage") {
            if (argc < 7) {
                print_usage();
                return 2;
            }
            const auto opts = parse_opts(argc, argv, 7);
            return run_wsvt_stage_cmd(
                argv[2], argv[3], argv[4], argv[5], argv[6], opts);
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
