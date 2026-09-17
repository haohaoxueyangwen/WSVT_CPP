#include "wsvt/directional_darkfield.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#if defined(WSVT_HAS_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace wsvt {
namespace {
constexpr double eps=std::numeric_limits<double>::epsilon();
std::size_t volume(std::size_t f,std::size_t h,std::size_t w) {
    const auto max=std::numeric_limits<std::size_t>::max();
    if(!f||!h||!w||h>max/w||f>max/(h*w)) throw std::invalid_argument("DDF invalid/overflowing shape");
    return f*h*w;
}
double modulo_pi(double a) {
    a=std::fmod(a,std::numbers::pi);
    return a<0 ? a+std::numbers::pi : a;
}
std::array<double,4> design(std::size_t y,std::size_t x,std::size_t h,std::size_t w) {
    auto freq=[](std::size_t k,std::size_t n) {
        const auto signed_k=k<(n+1)/2 ? static_cast<double>(k) : static_cast<double>(k)-static_cast<double>(n);
        return 2*std::numbers::pi*signed_k/static_cast<double>(n);
    };
    const double ky=freq(y,h),kx=freq(x,w);
    return {1,.5*kx*kx,kx*ky,.5*ky*ky};
}
template<class T> void require_finite(std::span<const T> data) {
    for(auto v:data) if(!std::isfinite(v)) throw std::invalid_argument("DDF input contains NaN/Inf");
}

#if defined(WSVT_HAS_OPENCV)
struct PatchWorkspace {
    cv::Mat spatial,frequency;
    std::vector<double> window,spectrum_s,spectrum_r;
    PatchWorkspace(int n,std::size_t f,bool hamming)
        : spatial(n,n,CV_64F),frequency(n,n,CV_64FC2),
          window(static_cast<std::size_t>(n)*n),spectrum_s(f*window.size()),spectrum_r(spectrum_s.size()) {
        for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            auto h=[&](int i){ return hamming ? .54-.46*std::cos(2*std::numbers::pi*i/(n-1)) : 1.; };
            window[static_cast<std::size_t>(y)*n+x]=h(y)*h(x);
        }
    }
    void transform(std::span<const double> raw,std::vector<double>& amplitude,std::size_t frames,int n) {
        const auto plane=window.size();
        for(std::size_t f=0;f<frames;++f) {
            auto* dst=spatial.ptr<double>();
            for(std::size_t i=0;i<plane;++i) dst[i]=raw[f*plane+i]*window[i];
            cv::dft(spatial,frequency,cv::DFT_COMPLEX_OUTPUT);
            const auto* src=frequency.ptr<cv::Vec2d>();
            for(std::size_t i=0;i<plane;++i) amplitude[f*plane+i]=std::hypot(src[i][0],src[i][1]);
        }
        (void)n;
    }
};

bool texture_2d(std::span<const double> reference,std::size_t frames,int n) {
    double xx=0,xy=0,yy=0;
    for(std::size_t f=0;f<frames;++f) for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
        auto at=[&](int j,int i){return reference[(f*static_cast<std::size_t>(n)+j)*n+i];};
        const double gx=x==0 ? at(y,1)-at(y,0) :
            x==n-1 ? at(y,n-1)-at(y,n-2) : .5*(at(y,x+1)-at(y,x-1));
        const double gy=y==0 ? at(1,x)-at(0,x) :
            y==n-1 ? at(n-1,x)-at(n-2,x) : .5*(at(y+1,x)-at(y-1,x));
        xx+=gx*gx;xy+=gx*gy;yy+=gy*gy;
    }
    const double radius=std::hypot(xx-yy,2*xy);
    const double lo=.5*(xx+yy-radius),hi=.5*(xx+yy+radius);
    return std::isfinite(hi)&&hi>0&&lo>256*eps*hi;
}

DdfFit patch_impl(std::span<const double> s,std::span<const double> r,
                  std::size_t f,const DdfConfig& cfg,PatchWorkspace& workspace) {
    if(!texture_2d(r,f,cfg.window)) {
        DdfFit out;out.status=DdfStatus::InsufficientTexture;return out;
    }
    workspace.transform(s,workspace.spectrum_s,f,cfg.window);
    workspace.transform(r,workspace.spectrum_r,f,cfg.window);
    auto out=fit_directional_spectra(workspace.spectrum_s,workspace.spectrum_r,
                                     f,static_cast<std::size_t>(cfg.window),static_cast<std::size_t>(cfg.window),cfg);
    out.fft_calls=2*f;
    return out;
}
#endif
}

const char* ddf_status_name(DdfStatus status) {
    switch(status) {
    case DdfStatus::Ok:return "ok";
    case DdfStatus::InsufficientSpectrum:return "insufficient_spectrum";
    case DdfStatus::IllConditioned:return "ill_conditioned";
    case DdfStatus::NonfiniteModel:return "nonfinite_model";
    case DdfStatus::InsufficientTexture:return "insufficient_2d_texture";
    case DdfStatus::NonPsd:return "non_psd";
    case DdfStatus::ModelMismatch:return "model_mismatch";
    case DdfStatus::OutsideDomain:return "outside_valid_domain";
    case DdfStatus::OutsideDisplacement:return "outside_displacement_map";
    case DdfStatus::InvalidDisplacement:return "invalid_displacement";
    }
    throw std::invalid_argument("unknown DDF status");
}
bool directional_darkfield_available() noexcept {
#if defined(WSVT_HAS_OPENCV)
    return true;
#else
    return false;
#endif
}
const char* directional_darkfield_backend() noexcept {
#if defined(WSVT_HAS_OPENCV)
    return "OpenCV " CV_VERSION " double DFT/SVD";
#else
    return "unavailable: build with OpenCV";
#endif
}
void DdfConfig::validate() const {
    if(window<4||stride<1||threads<1||threads>1024||(interpolation_order!=0&&interpolation_order!=1)||
       !std::isfinite(max_condition)||max_condition<1||
       !std::isfinite(max_model_residual)||max_model_residual<=0||
       !std::isfinite(min_orientation_anisotropy)||min_orientation_anisotropy<0||min_orientation_anisotropy>1)
        throw std::invalid_argument("invalid DDF window/stride/threads/interpolation/QC configuration");
    const bool missing_p=std::isnan(pixel_size_m),missing_z=std::isnan(distance_m);
    if(missing_p!=missing_z||(!missing_p&&(!std::isfinite(pixel_size_m)||!std::isfinite(distance_m)||
                                         pixel_size_m<=0||distance_m<=0)))
        throw std::invalid_argument("DDF angular units require finite positive pixel size and distance");
}

DdfFit ddf_tensor_observables(double xx,double xy,double yy,const DdfConfig& cfg) {
    cfg.validate();
    if(!std::isfinite(xx)||!std::isfinite(xy)||!std::isfinite(yy))
        throw std::invalid_argument("DDF tensor must be finite");
    DdfFit out;out.cov_xx_px2=xx;out.cov_xy_px2=xy;out.cov_yy_px2=yy;
    const double gap=std::hypot(xx-yy,2*xy),lo=.5*(xx+yy-gap),hi=.5*(xx+yy+gap);
    out.eigenvalue_min_raw_px2=lo;out.eigenvalue_max_raw_px2=hi;
    const double tol=256*eps*std::max({1.,std::abs(lo),std::abs(hi)});
    out.physical_valid=lo>=-tol;
    out.roundoff_psd_clipped=out.physical_valid&&lo<0;
    if(!out.physical_valid) {out.status=DdfStatus::NonPsd;return out;}
    const double l=std::max(lo,0.),h=std::max(hi,0.),sum=l+h;
    out.mean_scatter_variance_px2=sum/2;out.sigma_major_px=std::sqrt(h);out.sigma_minor_px=std::sqrt(l);
    out.anisotropy=sum>tol ? (h-l)/sum : 0;
    out.fractional_anisotropy_2d=sum>tol ? (h-l)/std::hypot(h,l) : 0;
    out.orientation_valid=sum>tol&&h-l>tol&&out.anisotropy>=cfg.min_orientation_anisotropy;
    if(out.orientation_valid) {
        out.scattering_angle_rad=modulo_pi(.5*std::atan2(2*xy,xx-yy));
        out.minor_scattering_angle_rad=modulo_pi(out.scattering_angle_rad+std::numbers::pi/2);
    }
    return out;
}
double ddf_directional_visibility(double xx,double xy,double yy,double theta,double frequency) {
    if(!std::isfinite(theta)||!std::isfinite(frequency)||frequency<0||
       !ddf_tensor_observables(xx,xy,yy).physical_valid)
        throw std::invalid_argument("visibility requires PSD tensor and finite angle/frequency");
    const double c=std::cos(theta),s=std::sin(theta),k=2*std::numbers::pi*frequency;
    return std::exp(-.5*k*k*(xx*c*c+2*xy*c*s+yy*s*s));
}

DdfFit fit_directional_spectra(std::span<const double> sample,std::span<const double> reference,
                              std::size_t frames,std::size_t h,std::size_t w,const DdfConfig& cfg) {
    cfg.validate();
    const auto count=volume(frames,h,w);
    if(h<4||w<4||count!=sample.size()||count!=reference.size()||count>static_cast<std::size_t>(INT32_MAX))
        throw std::invalid_argument("DDF spectra must be paired F,H,W; bounded int32 observation count");
    require_finite(sample);require_finite(reference);
#if defined(WSVT_HAS_OPENCV)
    double sm=0,rm=0;
    for(std::size_t i=0;i<count;++i) {
        if(sample[i]<0||reference[i]<0) throw std::invalid_argument("negative spectral magnitude");
        sm=std::max(sm,sample[i]);rm=std::max(rm,reference[i]);
    }
    const double sfloor=64*eps*std::max(sm,std::numeric_limits<double>::min());
    const double rfloor=64*eps*std::max(rm,std::numeric_limits<double>::min());
    std::vector<std::size_t> selected;selected.reserve(count);
    double svmax=0,rvmax=0;
    for(std::size_t i=0;i<count;++i) if(sample[i]>sfloor&&reference[i]>rfloor) {
        selected.push_back(i);svmax=std::max(svmax,sample[i]);rvmax=std::max(rvmax,reference[i]);
    }
    DdfFit out;out.observations=selected.size();
    if(selected.size()<4) return out;
    cv::Mat a(static_cast<int>(selected.size()),4,CV_64F),b(static_cast<int>(selected.size()),1,CV_64F);
    for(std::size_t j=0;j<selected.size();++j) {
        const auto i=selected[j],within=i%(h*w);
        const auto q=design(within/w,within%w,h,w);
        const double weight=sample[i]/svmax;
        for(int k=0;k<4;++k) a.at<double>(static_cast<int>(j),k)=q[static_cast<std::size_t>(k)]*weight;
        b.at<double>(static_cast<int>(j))=(std::log(reference[i])-std::log(sample[i]))*weight;
    }
    cv::Mat singular,u,vt;
    cv::SVD::compute(a,singular,u,vt);
    out.svd_calls=1;
    const double smax=singular.at<double>(0),smin=singular.at<double>(3);
    const double rank_tolerance=eps*static_cast<double>(std::max<std::size_t>(selected.size(),4))*smax;
    for(int k=0;k<4;++k) out.rank+=singular.at<double>(k)>rank_tolerance ? 1 : 0;
    out.condition=smin>0 ? smax/smin : std::numeric_limits<double>::infinity();
    if(out.rank!=4||out.condition>cfg.max_condition) {out.status=DdfStatus::IllConditioned;return out;}
    std::array<double,4> coefficients{};
    for(int k=0;k<4;++k) {
        double projected=0;
        for(int j=0;j<a.rows;++j) projected+=u.at<double>(j,k)*b.at<double>(j);
        for(int j=0;j<4;++j) coefficients[static_cast<std::size_t>(j)]+=
            vt.at<double>(k,j)*projected/singular.at<double>(k);
    }
    for(double c:coefficients) if(!std::isfinite(c)) {out.status=DdfStatus::NonfiniteModel;return out;}
    if(std::abs(coefficients[0])>700) {out.status=DdfStatus::NonfiniteModel;return out;}
    double residual=0,observed_power=0,logres=0,weight_power=0;
    const double scale=std::max(svmax,rvmax);
    for(std::size_t j=0;j<selected.size();++j) {
        const auto i=selected[j],within=i%(h*w);
        const auto q=design(within/w,within%w,h,w);
        double exponent=0;
        for(int k=0;k<4;++k) exponent-=q[static_cast<std::size_t>(k)]*coefficients[static_cast<std::size_t>(k)];
        if(std::abs(exponent)>700) {out.status=DdfStatus::NonfiniteModel;return out;}
        const double observed=sample[i]/scale,predicted=std::exp(exponent)*(reference[i]/scale);
        residual+=(observed-predicted)*(observed-predicted);observed_power+=observed*observed;
        const double weight=sample[i]/svmax,delta=(-exponent-(std::log(reference[i])-std::log(sample[i])))*weight;
        logres+=delta*delta;weight_power+=weight*weight;
    }
    auto tensor=ddf_tensor_observables(coefficients[1],coefficients[2],coefficients[3],cfg);
    tensor.rank=out.rank;tensor.condition=out.condition;tensor.observations=out.observations;tensor.svd_calls=1;
    tensor.mu=coefficients[0];tensor.transmission=std::exp(-coefficients[0]);
    tensor.numerical_valid=true;tensor.model_residual=std::sqrt(residual/observed_power);
    tensor.log_residual=std::sqrt(logres/weight_power);
    tensor.fit_valid=tensor.physical_valid&&tensor.model_residual<=cfg.max_model_residual;
    tensor.orientation_valid=tensor.orientation_valid&&tensor.fit_valid;
    tensor.status=tensor.fit_valid ? DdfStatus::Ok : tensor.physical_valid ? DdfStatus::ModelMismatch : DdfStatus::NonPsd;
    if(!tensor.orientation_valid) tensor.scattering_angle_rad=tensor.minor_scattering_angle_rad=ddf_nan;
    return tensor;
#else
    throw std::runtime_error("directional dark-field requires OpenCV double DFT/SVD");
#endif
}

DdfFit fit_directional_patch(std::span<const double> s,std::span<const double> r,
                             std::size_t frames,const DdfConfig& cfg) {
    cfg.validate();
    const auto count=volume(frames,static_cast<std::size_t>(cfg.window),static_cast<std::size_t>(cfg.window));
    if(count!=s.size()||count!=r.size()||count>static_cast<std::size_t>(INT32_MAX)) throw std::invalid_argument("DDF patch shape");
    require_finite(s);require_finite(r);
#if defined(WSVT_HAS_OPENCV)
    PatchWorkspace workspace(cfg.window,frames,cfg.hamming);
    return patch_impl(s,r,frames,cfg,workspace);
#else
    throw std::runtime_error("directional dark-field requires OpenCV");
#endif
}

DdfMap analyze_directional_darkfield(std::span<const float> s,std::span<const float> r,
    std::size_t frames,std::size_t h,std::size_t w,const DdfConfig& cfg,const DdfDisplacementView* disp,
    bool allow_unregistered,std::span<const std::int64_t> requested_y,std::span<const std::int64_t> requested_x) {
    const auto begin=std::chrono::steady_clock::now();cfg.validate();
    const auto count=volume(frames,h,w);
    if(count!=s.size()||count!=r.size()||h<static_cast<std::size_t>(cfg.window)||w<static_cast<std::size_t>(cfg.window)||
       h>INT32_MAX||w>INT32_MAX) throw std::invalid_argument("DDF raw stack shape");
    require_finite(s);require_finite(r);
    if(!disp&&!allow_unregistered) throw std::invalid_argument("DDF requires displacement or explicit unregistered mode");
    if(disp&&(disp->x.size()!=volume(1,disp->h,disp->w)||disp->y.size()!=disp->x.size()||
              std::abs(static_cast<double>(disp->origin_y))>INT32_MAX||std::abs(static_cast<double>(disp->origin_x))>INT32_MAX))
        throw std::invalid_argument("DDF displacement shape/origin");
    const auto patch_count=volume(frames,static_cast<std::size_t>(cfg.window),static_cast<std::size_t>(cfg.window));
    if(patch_count>static_cast<std::size_t>(INT32_MAX)) throw std::invalid_argument("DDF patch observations exceed int32");
    DdfMap out;
    const auto left=cfg.window/2;
    if(requested_y.empty()) for(std::int64_t y=left;y<=static_cast<std::int64_t>(h)-(cfg.window-left);y+=cfg.stride) out.centers_y.push_back(y);
    else out.centers_y.assign(requested_y.begin(),requested_y.end());
    if(requested_x.empty()) for(std::int64_t x=left;x<=static_cast<std::int64_t>(w)-(cfg.window-left);x+=cfg.stride) out.centers_x.push_back(x);
    else out.centers_x.assign(requested_x.begin(),requested_x.end());
    for(auto v:out.centers_y) if(v<INT32_MIN||v>INT32_MAX) throw std::invalid_argument("DDF center overflow");
    for(auto v:out.centers_x) if(v<INT32_MIN||v>INT32_MAX) throw std::invalid_argument("DDF center overflow");
    out.fits.resize(volume(1,out.centers_y.size(),out.centers_x.size()));
#if defined(WSVT_HAS_OPENCV)
    std::exception_ptr failure;
    #pragma omp parallel num_threads(cfg.threads)
    {
        // Allocation failures and OpenCV exceptions must not escape OpenMP.
        std::unique_ptr<PatchWorkspace> workspace;
        std::vector<double> sp,rp;
        try {workspace=std::make_unique<PatchWorkspace>(cfg.window,frames,cfg.hamming);sp.resize(patch_count);rp.resize(patch_count);}
        catch(...) {
            workspace.reset();
            #pragma omp critical(wsvt_ddf_failure)
            {if(!failure) failure=std::current_exception();}
        }
        #pragma omp for schedule(static)
        for(std::size_t i=0;i<out.fits.size();++i) {
            if(!workspace) continue;
            try {
                auto& fit=out.fits[i];fit.status=DdfStatus::OutsideDomain;
                const auto y0=out.centers_y[i/out.centers_x.size()],x0=out.centers_x[i%out.centers_x.size()];
                const auto y=y0-left,x=x0-left;
                if(y<0||x<0||y+cfg.window>static_cast<std::int64_t>(h)||x+cfg.window>static_cast<std::int64_t>(w)) continue;
                double ux=0,uy=0;
                if(disp) {
                    const auto j=y0-disp->origin_y,k=x0-disp->origin_x;
                    if(j<0||k<0||j>=static_cast<std::int64_t>(disp->h)||k>=static_cast<std::int64_t>(disp->w)) {
                        fit.status=DdfStatus::OutsideDisplacement;continue;
                    }
                    const auto di=static_cast<std::size_t>(j)*disp->w+static_cast<std::size_t>(k);
                    ux=-disp->x[di];uy=-disp->y[di];
                    if(!std::isfinite(ux)||!std::isfinite(uy)) {fit.status=DdfStatus::InvalidDisplacement;continue;}
                }
                const double request_x=ux,request_y=uy;
                if(cfg.interpolation_order==0) {ux=std::floor(ux+.5);uy=std::floor(uy+.5);}
                if(y+uy<0||x+ux<0||y+uy+cfg.window-1>static_cast<double>(h-1)||x+ux+cfg.window-1>static_cast<double>(w-1)) continue;
                for(std::size_t f=0;f<frames;++f) for(int py=0;py<cfg.window;++py) for(int px=0;px<cfg.window;++px) {
                    const auto pi=(f*static_cast<std::size_t>(cfg.window)+static_cast<std::size_t>(py))*cfg.window+static_cast<std::size_t>(px);
                    sp[pi]=s[(f*h+static_cast<std::size_t>(y+py))*w+static_cast<std::size_t>(x+px)];
                    const double ry=y+py+uy,rx=x+px+ux;
                    const auto iy=static_cast<std::size_t>(std::floor(ry)),ix=static_cast<std::size_t>(std::floor(rx));
                    const auto jy=std::min(iy+1,h-1),jx=std::min(ix+1,w-1);
                    const double dy=ry-static_cast<double>(iy),dx=rx-static_cast<double>(ix);
                    const auto at=[&](std::size_t yy,std::size_t xx){return static_cast<double>(r[(f*h+yy)*w+xx]);};
                    rp[pi]=(1-dy)*((1-dx)*at(iy,ix)+dx*at(iy,jx))+dy*((1-dx)*at(jy,ix)+dx*at(jy,jx));
                }
                fit=patch_impl(sp,rp,frames,cfg,*workspace);
                fit.reference_lookup_x=ux;fit.reference_lookup_y=uy;
                fit.requested_reference_lookup_x=request_x;fit.requested_reference_lookup_y=request_y;
                if(!std::isnan(cfg.pixel_size_m)) {
                    const double factor=std::pow(cfg.pixel_size_m/cfg.distance_m,2);
                    fit.cov_xx_rad2=fit.cov_xx_px2*factor;fit.cov_xy_rad2=fit.cov_xy_px2*factor;fit.cov_yy_rad2=fit.cov_yy_px2*factor;
                }
            } catch(...) {
                #pragma omp critical(wsvt_ddf_failure)
                {if(!failure) failure=std::current_exception();}
            }
        }
    }
    if(failure) std::rethrow_exception(failure);
    for(const auto& fit:out.fits) {out.fft_calls+=fit.fft_calls;out.svd_calls+=fit.svd_calls;}
    out.analysis_time_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    return out;
#else
    throw std::runtime_error("directional dark-field requires OpenCV");
#endif
}
}
