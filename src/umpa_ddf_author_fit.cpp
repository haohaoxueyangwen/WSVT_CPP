// SPDX-License-Identifier: GPL-3.0-or-later
// Port of Smith et al. (2022) S1 single_model.py and utils.py.
// See docs/UMPA_DDF_AUTHOR_PORT.md for provenance.
#include "wsvt/umpa_ddf_author_fit.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace wsvt {
namespace {
using State=std::array<double,3>;
struct Vertex { State x; UmpaDdfKernelParameters kernel; UmpaDdfKernelFit fit;
    double penalty=0., f=0.; };
void transform(Vertex& v,const UmpaDdfAuthorSettings& s) {
    double eps=v.x[0]*v.x[0]+v.x[1]*v.x[1];
    const double theta=std::atan2(v.x[1],v.x[0]);
    double sig=v.x[2];
    if(eps>=1.) {v.penalty=eps-1.;eps=eps/(eps+0.01);}
    // Preserve author's literal 10, including when max_sigma differs.
    if(sig>s.max_sigma) v.penalty+=(sig-10.)*10.;
    else if(sig<0.) {v.penalty+=sig*-10.;sig=0.;}
    const double xx=sig*sig/(1.+eps)+s.blur_extra*s.blur_extra;
    const double yy=sig*sig/(1.-eps)+s.blur_extra*s.blur_extra;
    const double sn=std::sin(theta),cs=std::cos(theta),sn2=std::sin(2.*theta);
    v.kernel={8U,0.5*(cs*cs/xx+sn*sn/yy),0.5*(sn2/xx-sn2/yy),0.5*(sn*sn/xx+cs*cs/yy)};
}
}
UmpaDdfAuthorResult fit_umpa_ddf_author_fixed_u(
    std::span<const double> sample,std::span<const double> reference,
    std::size_t frames,std::size_t height,std::size_t width,
    std::size_t y,std::size_t x,std::ptrdiff_t uy,std::ptrdiff_t ux,
    std::size_t window_half,std::ptrdiff_t max_shift,UmpaDdfAuthorSettings settings) {
    if(settings.max_iterations==0 || settings.max_iterations>100000 ||
       !std::isfinite(settings.blur_extra) || settings.blur_extra<=0. ||
       !std::isfinite(settings.max_sigma) || settings.max_sigma<=0.)
        throw std::invalid_argument("invalid author fit settings");
    for(double q:settings.initial) if(!std::isfinite(q))
        throw std::invalid_argument("nonfinite initial transform");
    std::size_t calls=0;
    auto evaluate=[&](State state) {
        Vertex v;v.x=state;transform(v,settings);
        v.fit=fit_umpa_ddf_kernel_official_integer_at(sample,reference,frames,height,width,
            y,x,uy,ux,window_half,max_shift,v.kernel);
        if(!v.fit.numerical_valid) throw std::runtime_error("invalid dark-field observations");
        v.f=v.fit.cost+v.penalty;++calls;return v;
    };
    std::array<Vertex,4> sim;
    sim[0]=evaluate(settings.initial);
    for(std::size_t k=0;k<3;++k) {auto z=settings.initial;
        z[k]=z[k]!=0.?1.05*z[k]:0.00025;sim[k+1]=evaluate(z);}
    auto sort=[&](){std::stable_sort(sim.begin(),sim.end(),
        [](const Vertex& a,const Vertex& b){return a.f<b.f;});};
    sort();std::size_t iterations=1;bool converged=false;
    // S1 calls scipy Nelder-Mead: adaptive=False, xatol=1e-4, fatol=1e-12.
    // With maxiter supplied and maxfev omitted scipy leaves maxfev unlimited.
    while(iterations<settings.max_iterations) {
        double dx=0.,df=0.;
        for(std::size_t i=1;i<4;++i) {
            df=std::max(df,std::abs(sim[0].f-sim[i].f));
            for(std::size_t j=0;j<3;++j) dx=std::max(dx,std::abs(sim[i].x[j]-sim[0].x[j]));
        }
        if(dx<=1e-4 && df<=1e-12) {converged=true;break;}
        State center{};
        for(std::size_t j=0;j<3;++j) center[j]=(sim[0].x[j]+sim[1].x[j]+sim[2].x[j])/3.;
        auto candidate=[&](double a,double b) {State z;
            for(std::size_t j=0;j<3;++j) z[j]=a*center[j]+b*sim[3].x[j];
            return evaluate(z);};
        auto reflected=candidate(2.,-1.);bool shrink=false;
        if(reflected.f<sim[0].f) {auto expanded=candidate(3.,-2.);
            sim[3]=expanded.f<reflected.f?expanded:reflected;
        } else if(reflected.f<sim[2].f) sim[3]=reflected;
        else if(reflected.f<sim[3].f) {auto contracted=candidate(1.5,-0.5);
            if(contracted.f<=reflected.f) sim[3]=contracted;else shrink=true;
        } else {auto contracted=candidate(0.5,0.5);
            if(contracted.f<sim[3].f) sim[3]=contracted;else shrink=true;
        }
        if(shrink) for(std::size_t i=1;i<4;++i) {State z;
            for(std::size_t j=0;j<3;++j) z[j]=sim[0].x[j]+0.5*(sim[i].x[j]-sim[0].x[j]);
            sim[i]=evaluate(z);}
        ++iterations;sort();
    }
    return {sim[0].x,sim[0].kernel,sim[0].fit,sim[0].penalty,iterations,calls,converged};
}
}
