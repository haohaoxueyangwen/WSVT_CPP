#include "wsvt/directional_darkfield.hpp"
#include "wsvt/io_h5.hpp"
#include "wsvt/io_json.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace wsvt;
struct Field {const char* name;double DdfFit::*member;};
const std::vector<Field> fields{
    {"mu",&DdfFit::mu},{"transmission",&DdfFit::transmission},
    {"cov_xx_px2",&DdfFit::cov_xx_px2},{"cov_xy_px2",&DdfFit::cov_xy_px2},{"cov_yy_px2",&DdfFit::cov_yy_px2},
    {"eigenvalue_min_raw_px2",&DdfFit::eigenvalue_min_raw_px2},{"eigenvalue_max_raw_px2",&DdfFit::eigenvalue_max_raw_px2},
    {"mean_scatter_variance_px2",&DdfFit::mean_scatter_variance_px2},
    {"sigma_major_px",&DdfFit::sigma_major_px},{"sigma_minor_px",&DdfFit::sigma_minor_px},
    {"anisotropy",&DdfFit::anisotropy},{"fractional_anisotropy_2d",&DdfFit::fractional_anisotropy_2d},
    {"scattering_angle_rad",&DdfFit::scattering_angle_rad},{"minor_scattering_angle_rad",&DdfFit::minor_scattering_angle_rad},
    {"condition",&DdfFit::condition},{"model_residual",&DdfFit::model_residual},{"log_residual",&DdfFit::log_residual},
    {"cov_xx_rad2",&DdfFit::cov_xx_rad2},{"cov_xy_rad2",&DdfFit::cov_xy_rad2},{"cov_yy_rad2",&DdfFit::cov_yy_rad2},
    {"reference_lookup_x",&DdfFit::reference_lookup_x},{"reference_lookup_y",&DdfFit::reference_lookup_y},
    {"requested_reference_lookup_x",&DdfFit::requested_reference_lookup_x},{"requested_reference_lookup_y",&DdfFit::requested_reference_lookup_y}
};
std::int64_t integer(const std::string& s) {
    std::size_t n=0;const auto result=std::stoll(s,&n);
    if(n!=s.size()||result<INT32_MIN||result>INT32_MAX) throw std::invalid_argument("invalid/bounded integer: "+s);
    return result;
}
double real(const std::string& s) {
    std::size_t n=0;const double result=std::stod(s,&n);
    if(n!=s.size()||!std::isfinite(result)) throw std::invalid_argument("invalid finite number: "+s);
    return result;
}
std::vector<std::int64_t> coordinates(const std::string& s) {
    std::vector<std::int64_t> out;std::istringstream in(s);std::string item;
    while(std::getline(in,item,',')) out.push_back(integer(item));
    if(out.empty()||s.back()==',') throw std::invalid_argument("empty coordinate list");
    return out;
}
JsonValue finite_json(double value) {return std::isfinite(value) ? JsonValue(value) : JsonValue(nullptr);}
void usage() {
    std::cout<<"Native C++ directional dark-field (raw intensity Fourier tensor fit)\n"
      <<"Usage: wsvt_ddf_cli INPUT.h5 NEW_OUTPUT_DIR [options]\n"
      <<"  --sample-key sample --reference-key ref\n"
      <<"  --wsvt-result DISPLACEMENT.h5 --origin-y Y --origin-x X\n"
      <<"  --allow-unregistered     (alternative to a displacement file)\n"
      <<"  --raw-origin-y Y --raw-origin-x X   (default 0; raw ROI global coordinates)\n"
      <<"  --window 16 --stride 8 --threads 4 --apodization hamming|rectangular\n"
      <<"  --interpolation-order 0|1 (0 default; linear resampling can bias blur)\n"
      <<"  --centers-y 32,40,48 --centers-x 32,40,48  (optional raw-input-local centers)\n"
      <<"  --max-condition 1e8 --max-model-residual 0.15 --min-orientation-anisotropy 0.1\n"
      <<"  --pixel-size-m P --distance-m Z     (both required for angular units)\n"
      <<"Output: directional_darkfield.hdf5 (float32), CSV (double), params/bench_report JSON.\n";
}
}

int main(int argc,char** argv) try {
    const auto start=std::chrono::steady_clock::now();
    if(argc==2&&(std::string(argv[1])=="--help"||std::string(argv[1])=="-h")) {usage();return 0;}
    if(argc<3) {usage();return 2;}
    const std::filesystem::path input=argv[1],output=argv[2];
    if(std::filesystem::exists(output)) throw std::invalid_argument("output directory already exists; refusing overwrite");
    if(!directional_darkfield_available()) throw std::runtime_error(directional_darkfield_backend());
    DdfConfig cfg;std::string sample_key="sample",reference_key="ref",displacement_file;
    std::int64_t origin_y=0,origin_x=0,raw_y=0,raw_x=0;
    bool has_oy=false,has_ox=false,unregistered=false;
    std::vector<std::int64_t> cy,cx;
    std::map<std::string,bool> seen;
    for(int i=3;i<argc;++i) {
        const std::string key=argv[i];
        if(!seen.emplace(key,true).second) throw std::invalid_argument("duplicate option: "+key);
        if(key=="--allow-unregistered") {unregistered=true;continue;}
        if(i+1==argc) throw std::invalid_argument("missing option value: "+key);
        const std::string value=argv[++i];
        if(key=="--sample-key") sample_key=value;
        else if(key=="--reference-key") reference_key=value;
        else if(key=="--wsvt-result") displacement_file=value;
        else if(key=="--origin-y") {origin_y=integer(value);has_oy=true;}
        else if(key=="--origin-x") {origin_x=integer(value);has_ox=true;}
        else if(key=="--raw-origin-y") raw_y=integer(value);
        else if(key=="--raw-origin-x") raw_x=integer(value);
        else if(key=="--window") cfg.window=static_cast<int>(integer(value));
        else if(key=="--stride") cfg.stride=static_cast<int>(integer(value));
        else if(key=="--threads") cfg.threads=static_cast<int>(integer(value));
        else if(key=="--interpolation-order") cfg.interpolation_order=static_cast<int>(integer(value));
        else if(key=="--apodization") {
            if(value!="hamming"&&value!="rectangular") throw std::invalid_argument("invalid apodization");
            cfg.hamming=value=="hamming";
        }
        else if(key=="--max-condition") cfg.max_condition=real(value);
        else if(key=="--max-model-residual") cfg.max_model_residual=real(value);
        else if(key=="--min-orientation-anisotropy") cfg.min_orientation_anisotropy=real(value);
        else if(key=="--pixel-size-m") cfg.pixel_size_m=real(value);
        else if(key=="--distance-m") cfg.distance_m=real(value);
        else if(key=="--centers-y") cy=coordinates(value);
        else if(key=="--centers-x") cx=coordinates(value);
        else throw std::invalid_argument("unknown option: "+key);
    }
    cfg.validate();
    if(displacement_file.empty()) {
        if(!unregistered||has_oy||has_ox) throw std::invalid_argument("provide displacement with origins, or explicit unregistered mode");
    } else if(!has_oy||!has_ox||unregistered)
        throw std::invalid_argument("external displacement requires both explicit origins and no unregistered flag");
    auto s=read_h5(input.string(),sample_key),r=read_h5(input.string(),reference_key);
    if(s.shape.size()!=3||s.shape!=r.shape) throw std::invalid_argument("sample/ref HDF5 shapes must match F,H,W");
    NdArrayF32 dx,dy;DdfDisplacementView disp;
    if(!displacement_file.empty()) {
        dx=read_h5(displacement_file,"displace_x");dy=read_h5(displacement_file,"displace_y");
        if(dx.shape.size()!=2||dx.shape!=dy.shape) throw std::invalid_argument("displacement HDF5 fields must match H,W");
        disp={dx.data,dy.data,dx.shape[0],dx.shape[1],origin_y-raw_y,origin_x-raw_x};
    }
    const auto result=analyze_directional_darkfield(s.data,r.data,s.shape[0],s.shape[1],s.shape[2],cfg,
        displacement_file.empty()?nullptr:&disp,unregistered,cy,cx);
    if(!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    if(!std::filesystem::create_directory(output)) throw std::runtime_error("output directory was created concurrently");
    const auto ny=result.centers_y.size(),nx=result.centers_x.size();
    std::ofstream csv(output/"directional_darkfield.csv");csv<<std::setprecision(17);
    csv<<"raw_y,raw_x";
    for(const auto& field:fields) csv<<','<<field.name;
    csv<<",rank,observations,numerical_valid,physical_valid,fit_valid,orientation_valid,roundoff_psd_clipped,status\n";
    std::vector<H5ItemF32> datasets;
    for(const auto& field:fields) {
        std::vector<float> values;values.reserve(result.fits.size());
        for(const auto& f:result.fits) values.push_back(static_cast<float>(f.*field.member));
        datasets.push_back({field.name,{{ny,nx},std::move(values)}});
    }
    auto dataset=[&](const char* name,auto getter) {
        std::vector<float> values;values.reserve(result.fits.size());
        for(const auto& f:result.fits) values.push_back(static_cast<float>(getter(f)));
        datasets.push_back({name,{{ny,nx},std::move(values)}});
    };
    dataset("rank",[](const DdfFit& f){return f.rank;});
    dataset("observations",[](const DdfFit& f){return f.observations;});
    dataset("numerical_valid",[](const DdfFit& f){return f.numerical_valid;});
    dataset("physical_valid",[](const DdfFit& f){return f.physical_valid;});
    dataset("fit_valid",[](const DdfFit& f){return f.fit_valid;});
    dataset("orientation_valid",[](const DdfFit& f){return f.orientation_valid;});
    dataset("roundoff_psd_clipped",[](const DdfFit& f){return f.roundoff_psd_clipped;});
    dataset("status_code",[](const DdfFit& f){return static_cast<int>(f.status);});
    std::vector<float> global_y,global_x;
    for(auto y:result.centers_y) global_y.push_back(static_cast<float>(y+raw_y));
    for(auto x:result.centers_x) global_x.push_back(static_cast<float>(x+raw_x));
    datasets.push_back({"centers_raw_y",{{ny},std::move(global_y)}});
    datasets.push_back({"centers_raw_x",{{nx},std::move(global_x)}});
    JsonArray exact_y,exact_x;
    for(auto y:result.centers_y) exact_y.emplace_back(static_cast<double>(y+raw_y));
    for(auto x:result.centers_x) exact_x.emplace_back(static_cast<double>(x+raw_x));
    std::map<std::string,std::size_t> counts;
    std::size_t valid=0,oriented=0;
    for(std::size_t i=0;i<result.fits.size();++i) {
        const auto& f=result.fits[i];
        csv<<result.centers_y[i/nx]+raw_y<<','<<result.centers_x[i%nx]+raw_x;
        for(const auto& field:fields) csv<<','<<f.*field.member;
        csv<<','<<f.rank<<','<<f.observations<<','<<f.numerical_valid<<','<<f.physical_valid<<','<<f.fit_valid
           <<','<<f.orientation_valid<<','<<f.roundoff_psd_clipped<<','<<ddf_status_name(f.status)<<'\n';
        ++counts[ddf_status_name(f.status)];valid+=f.fit_valid;oriented+=f.orientation_valid;
    }
    csv.close();if(!csv) throw std::runtime_error("CSV write failed");
    write_h5(output.string(),"directional_darkfield",datasets,0);
    JsonObject codes,status_counts;
    for(int i=0;i<=static_cast<int>(DdfStatus::InvalidDisplacement);++i)
        codes[ddf_status_name(static_cast<DdfStatus>(i))]=i;
    for(const auto& [name,count]:counts) status_counts[name]=static_cast<double>(count);
    JsonObject params{
        {"method","native C++ Lautizi2024 Eq3 weighted Fourier tensor"},{"backend",directional_darkfield_backend()},
        {"input",std::filesystem::absolute(input).string()},{"sample_key",sample_key},{"reference_key",reference_key},
        {"input_conversion","existing HDF5 loader converts to float32; DDF computes in float64; CSV double/HDF5 float32"},
        {"wsvt_result",displacement_file},{"registered",!displacement_file.empty()},
        {"displacement_origin_y",static_cast<double>(origin_y)},{"displacement_origin_x",static_cast<double>(origin_x)},
        {"raw_origin_y",static_cast<double>(raw_y)},{"raw_origin_x",static_cast<double>(raw_x)},
        {"centers_raw_y",exact_y},{"centers_raw_x",exact_x},{"status_codes",codes},
        {"frames",static_cast<double>(s.shape[0])},{"input_h",static_cast<double>(s.shape[1])},{"input_w",static_cast<double>(s.shape[2])},
        {"window",cfg.window},{"stride",cfg.stride},{"threads",cfg.threads},
        {"apodization",cfg.hamming?"hamming":"rectangular"},{"interpolation_order",cfg.interpolation_order},
        {"max_condition",cfg.max_condition},{"max_model_residual",cfg.max_model_residual},
        {"min_orientation_anisotropy",cfg.min_orientation_anisotropy},
        {"pixel_size_m",finite_json(cfg.pixel_size_m)},{"distance_m",finite_json(cfg.distance_m)},
        {"covariance_units","pixel^2"},{"angle_convention","major scattering axis, modulo pi; x columns, y rows"},
        {"angular_geometry",std::isnan(cfg.pixel_size_m)?"not supplied":"parallel-beam small-angle"},
        {"physical_accuracy_claim",false}
    };
    write_json(output.string(),"params",params);
    const double e2e=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    JsonObject report{
        {"status","completed"},{"native_cpp",true},{"python_runtime_required",false},
        {"sampled_points",static_cast<double>(result.fits.size())},{"status_counts",status_counts},
        {"fit_valid_count",static_cast<double>(valid)},{"orientation_valid_count",static_cast<double>(oriented)},
        {"fft_calls",static_cast<double>(result.fft_calls)},{"svd_calls",static_cast<double>(result.svd_calls)},
        {"ddf_analysis_time_s_diagnostic",result.analysis_time_s},{"end_to_end_before_report_s_diagnostic",e2e},
        {"formal_timing",false},{"physical_accuracy_claim",false}
    };
    write_json(output.string(),"bench_report",report);
    std::cout<<json_dumps(JsonValue(report))<<'\n';
    return 0;
} catch(const std::exception& error) {std::cerr<<"DDF error: "<<error.what()<<'\n';return 1;}
