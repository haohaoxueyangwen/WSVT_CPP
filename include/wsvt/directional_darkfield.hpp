#pragma once
#include "wsvt/export.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace wsvt {
inline constexpr double ddf_nan=std::numeric_limits<double>::quiet_NaN();
enum class DdfStatus {
    Ok, InsufficientSpectrum, IllConditioned, NonfiniteModel,
    InsufficientTexture, NonPsd, ModelMismatch, OutsideDomain,
    OutsideDisplacement, InvalidDisplacement
};
WSVT_API const char* ddf_status_name(DdfStatus status);
WSVT_API bool directional_darkfield_available() noexcept;
WSVT_API const char* directional_darkfield_backend() noexcept;

struct DdfConfig {
    int window=16, stride=8, threads=1;
    bool hamming=true;
    int interpolation_order=0;
    double max_condition=1e8, max_model_residual=.15, min_orientation_anisotropy=.1;
    // Both NaN means no angular-unit conversion; both positive enables it.
    double pixel_size_m=ddf_nan, distance_m=ddf_nan;
    WSVT_API void validate() const;
};

struct DdfFit {
    double mu=ddf_nan, transmission=ddf_nan;
    double cov_xx_px2=ddf_nan,cov_xy_px2=ddf_nan,cov_yy_px2=ddf_nan;
    double eigenvalue_min_raw_px2=ddf_nan,eigenvalue_max_raw_px2=ddf_nan;
    double mean_scatter_variance_px2=ddf_nan,sigma_major_px=ddf_nan,sigma_minor_px=ddf_nan;
    double anisotropy=ddf_nan,fractional_anisotropy_2d=ddf_nan;
    double scattering_angle_rad=ddf_nan,minor_scattering_angle_rad=ddf_nan;
    double condition=std::numeric_limits<double>::infinity();
    double model_residual=ddf_nan,log_residual=ddf_nan;
    double cov_xx_rad2=ddf_nan,cov_xy_rad2=ddf_nan,cov_yy_rad2=ddf_nan;
    double requested_reference_lookup_x=ddf_nan,requested_reference_lookup_y=ddf_nan;
    double reference_lookup_x=ddf_nan,reference_lookup_y=ddf_nan;
    std::size_t observations=0;
    int rank=0;
    bool numerical_valid=false,physical_valid=false,fit_valid=false,orientation_valid=false;
    bool roundoff_psd_clipped=false;
    DdfStatus status=DdfStatus::InsufficientSpectrum;
    std::uint64_t fft_calls=0,svd_calls=0;
};

struct DdfDisplacementView {
    std::span<const float> x,y;
    std::size_t h=0,w=0;
    // Coordinates relative to raw stack [0,0], not an inferred crop.
    std::int64_t origin_y=0,origin_x=0;
};
struct DdfMap {
    std::vector<std::int64_t> centers_y,centers_x;
    std::vector<DdfFit> fits; // y-major sampled grid; never interpolated.
    std::uint64_t fft_calls=0,svd_calls=0;
    double analysis_time_s=0;
};

WSVT_API DdfFit ddf_tensor_observables(double xx,double xy,double yy,const DdfConfig& config={});
WSVT_API double ddf_directional_visibility(double xx,double xy,double yy,double angle_rad,double cycles_per_pixel);
/// Direct weighted SVD fit of full FFT magnitudes (F,H,W) in double precision.
WSVT_API DdfFit fit_directional_spectra(std::span<const double> sample_amplitude,
    std::span<const double> reference_amplitude,std::size_t frames,std::size_t h,
    std::size_t w,const DdfConfig& config={});
/// Reference entry for one raw F*window*window patch pair.
WSVT_API DdfFit fit_directional_patch(std::span<const double> sample,
    std::span<const double> reference,std::size_t frames,const DdfConfig& config={});
/// All computation is native C++; Python is not used or launched.
WSVT_API DdfMap analyze_directional_darkfield(std::span<const float> sample,
    std::span<const float> reference,std::size_t frames,std::size_t h,std::size_t w,
    const DdfConfig& config={},const DdfDisplacementView* displacement=nullptr,
    bool allow_unregistered=false,std::span<const std::int64_t> centers_y={},
    std::span<const std::int64_t> centers_x={});
}
