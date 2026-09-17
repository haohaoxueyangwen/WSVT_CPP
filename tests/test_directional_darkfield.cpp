#include "wsvt/directional_darkfield.hpp"
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace wsvt;
namespace {
void backend() { if(!directional_darkfield_available()) SKIP("OpenCV disabled: native DDF reports unavailable"); }
std::array<double,3> tensor(double theta) {
    const double c=std::cos(theta),s=std::sin(theta);
    return {.3+1.1*c*c,1.1*c*s,.3+1.1*s*s};
}
std::pair<std::vector<double>,std::vector<double>> spectra(int n,double xx,double xy,double yy,double mu) {
    std::vector<double> a(static_cast<std::size_t>(3*n*n)),b(a.size());
    for(std::size_t i=0;i<a.size();++i) {
        const auto x=static_cast<int>(i%static_cast<std::size_t>(n)),y=static_cast<int>((i/static_cast<std::size_t>(n))%static_cast<std::size_t>(n));
        const auto freq=[&](int k){return 2*std::numbers::pi*(k<(n+1)/2?k:k-n)/n;};
        const double kx=freq(x),ky=freq(y);
        b[i]=.5+static_cast<double>(i%37)/37;
        a[i]=b[i]*std::exp(-mu-.5*(xx*kx*kx+2*xy*kx*ky+yy*ky*ky));
    }
    return {a,b};
}
}
TEST_CASE("DDF exact spectra recover tensor and orientation", "[ddf]") {
    backend();
    for(int n:{16,17}) for(double degrees:{0.,30.,60.,90.,150.}) {
        const double theta=degrees*std::numbers::pi/180;
        auto c=tensor(theta);auto [a,b]=spectra(n,c[0],c[1],c[2],-.1);
        const auto fit=fit_directional_spectra(a,b,3,static_cast<std::size_t>(n),static_cast<std::size_t>(n));
        REQUIRE(fit.fit_valid);REQUIRE(fit.orientation_valid);
        REQUIRE(std::abs(fit.cov_xx_px2-c[0])<1e-10);
        REQUIRE(std::abs(fit.cov_xy_px2-c[1])<1e-10);
        REQUIRE(std::abs(fit.cov_yy_px2-c[2])<1e-10);
        REQUIRE(std::abs(fit.transmission-std::exp(.1))<1e-10);
        REQUIRE(std::abs(std::remainder(fit.scattering_angle_rad-theta,std::numbers::pi))<1e-10);
    }
}
TEST_CASE("DDF undefined orientation and non PSD are explicit", "[ddf]") {
    for(double value:{0.,.6}) {
        const auto fit=ddf_tensor_observables(value,0,value);
        REQUIRE(fit.physical_valid);REQUIRE_FALSE(fit.orientation_valid);
        REQUIRE(std::isnan(fit.scattering_angle_rad));
    }
    const auto bad=ddf_tensor_observables(-.2,0,.5);
    REQUIRE_FALSE(bad.physical_valid);REQUIRE(bad.cov_xx_px2==-.2);
    REQUIRE(std::isnan(bad.sigma_major_px));
    REQUIRE(ddf_directional_visibility(1.4,0,.3,0,.1)<
            ddf_directional_visibility(1.4,0,.3,std::numbers::pi/2,.1));
    REQUIRE_THROWS(ddf_directional_visibility(-1,0,1,0,.1));
}
TEST_CASE("DDF rejects invalid inputs in Release too", "[ddf]") {
    DdfConfig cfg;cfg.window=0;REQUIRE_THROWS(cfg.validate());
    cfg={};cfg.threads=0;REQUIRE_THROWS(cfg.validate());
    cfg={};cfg.pixel_size_m=1e-6;REQUIRE_THROWS(cfg.validate());
    cfg={};cfg.max_condition=std::numeric_limits<double>::quiet_NaN();REQUIRE_THROWS(cfg.validate());
    std::vector<double> a(256,1);a[0]=ddf_nan;
    REQUIRE_THROWS(fit_directional_spectra(a,a,1,16,16));
    a[0]=std::numeric_limits<double>::infinity();
    REQUIRE_THROWS(fit_directional_patch(a,a,1));
    std::vector<float> raw(256,1);
    REQUIRE_THROWS(analyze_directional_darkfield(raw,raw,1,16,16));
    REQUIRE_THROWS(analyze_directional_darkfield(raw,raw,1,0,16));
}
TEST_CASE("DDF flat and one dimensional raw patterns reject artificial window texture", "[ddf]") {
    backend();
    std::vector<double> a(3*16*16,1);
    REQUIRE(fit_directional_patch(a,a,3).status==DdfStatus::InsufficientTexture);
    for(std::size_t i=0;i<a.size();++i) a[i]=2+std::sin(static_cast<double>(i%16));
    REQUIRE(fit_directional_patch(a,a,3).status==DdfStatus::InsufficientTexture);
    std::vector<double> spectrum(3*16*16,0);
    for(std::size_t i=0;i<spectrum.size();++i) if((i/16)%16==0) spectrum[i]=1;
    const auto fit=fit_directional_spectra(spectrum,spectrum,3,16,16);
    REQUIRE_FALSE(fit.numerical_valid);REQUIRE(fit.status==DdfStatus::IllConditioned);
}
TEST_CASE("DDF raw registration and worker count preserve results", "[ddf]") {
    backend();
    constexpr int f=8,h=48,w=56;
    std::vector<float> r(f*h*w),s(r.size());
    std::mt19937 gen(20260911);std::uniform_real_distribution<float> d(.2f,2.f);
    for(auto& v:r) v=d(gen);
    for(int frame=0;frame<f;++frame) for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        s[static_cast<std::size_t>((frame*h+y)*w+x)]=r[static_cast<std::size_t>((frame*h+(y+h-1)%h)*w+(x+2)%w)];
    std::vector<float> dx(32*40,-2),dy(dx.size(),1);
    DdfDisplacementView disp{dx,dy,32,40,8,8};
    DdfConfig cfg;
    const std::vector<std::int64_t> ys{0,16,24,32},xs{0,16,24,32};
    const auto single=analyze_directional_darkfield(s,r,f,h,w,cfg,&disp,false,ys,xs);
    cfg.threads=4;
    const auto parallel=analyze_directional_darkfield(s,r,f,h,w,cfg,&disp,false,ys,xs);
    REQUIRE(single.fft_calls==parallel.fft_calls);
    REQUIRE(single.svd_calls==parallel.svd_calls);
    for(std::size_t i=0;i<single.fits.size();++i) {
        const auto& a=single.fits[i];const auto& b=parallel.fits[i];
        REQUIRE(a.status==b.status);REQUIRE(a.orientation_valid==b.orientation_valid);
        if(a.numerical_valid) {
            REQUIRE(a.cov_xx_px2==b.cov_xx_px2);REQUIRE(a.cov_xy_px2==b.cov_xy_px2);
            REQUIRE(std::abs(a.transmission-1)<1e-10);
            REQUIRE(std::abs(a.cov_xx_px2)<1e-10);
        }
    }
    dx[16*40+16]=std::numeric_limits<float>::quiet_NaN();
    const auto invalid=analyze_directional_darkfield(s,r,f,h,w,cfg,&disp,false,ys,xs);
    REQUIRE(invalid.fits[2*xs.size()+2].status==DdfStatus::InvalidDisplacement);
}
TEST_CASE("DDF angular units and intensity scaling", "[ddf]") {
    backend();
    auto [a,b]=spectra(16,1.4,.2,.3,.2);
    const auto baseline=fit_directional_spectra(a,b,3,16,16);
    for(double scale:{1e-12,1e12}) {
        auto as=a,bs=b;
        for(auto& v:as)v*=scale;for(auto& v:bs)v*=scale;
        const auto fit=fit_directional_spectra(as,bs,3,16,16);
        REQUIRE(std::abs(fit.cov_xy_px2-baseline.cov_xy_px2)<1e-10);
    }
    std::vector<float> raw(4*32*32);
    for(std::size_t i=0;i<raw.size();++i) raw[i]=static_cast<float>(2+std::sin(static_cast<double>(i)*.7)+std::cos(static_cast<double>(i)*.13));
    DdfConfig cfg;cfg.pixel_size_m=1e-6;cfg.distance_m=.5;
    const auto out=analyze_directional_darkfield(raw,raw,4,32,32,cfg,nullptr,true);
    REQUIRE(out.fits[0].numerical_valid);
    REQUIRE(out.fits[0].cov_xx_rad2==out.fits[0].cov_xx_px2*4e-12);
}
TEST_CASE("DDF missing OpenCV is reported explicitly", "[ddf]") {
    if(directional_darkfield_available()) SUCCEED("native DFT/SVD available");
    else {
        std::vector<double> a(256,1);
        REQUIRE_THROWS(fit_directional_spectra(a,a,1,16,16));
    }
}
