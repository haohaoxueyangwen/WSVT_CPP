#include "wsvt/umpa_ddf_author_fit.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#if defined(_WIN32)
#define DDF_EXPORT __declspec(dllexport)
#else
#define DDF_EXPORT __attribute__((visibility("default")))
#endif
// Output: sx,sy,sigma,a,b,c,penalty,T,cost,residual,nit,nfev,converged.
extern "C" DDF_EXPORT int wsvt_ddf_author_fit(
    const double* sample,const double* reference,std::size_t count,
    std::size_t frames,std::size_t height,std::size_t width,
    std::size_t y,std::size_t x,double lookup_y,double lookup_x,
    std::size_t window_half,int max_shift,const double* initial,
    double blur_extra,double max_sigma,std::size_t maxiter,
    double* output,char* error,std::size_t error_size) noexcept {
    try {
        if(!sample || !reference || !initial || !output || !frames || !height || !width ||
           height>std::numeric_limits<std::size_t>::max()/width ||
           frames>std::numeric_limits<std::size_t>::max()/(height*width) || count!=frames*height*width)
            throw std::invalid_argument("invalid stack buffer");
        // Archived Cython uses libc round, including halfway away from zero.
        if(!std::isfinite(lookup_y) || !std::isfinite(lookup_x) || max_shift<=0 ||
           std::abs(lookup_y)>=max_shift || std::abs(lookup_x)>=max_shift)
            throw std::invalid_argument("invalid lookup displacement");
        wsvt::UmpaDdfAuthorSettings settings{{initial[0],initial[1],initial[2]},blur_extra,max_sigma,maxiter};
        auto r=wsvt::fit_umpa_ddf_author_fixed_u({sample,count},{reference,count},frames,height,width,y,x,
            static_cast<std::ptrdiff_t>(std::round(lookup_y)),static_cast<std::ptrdiff_t>(std::round(lookup_x)),
            window_half,max_shift,settings);
        const double values[]={r.transform[0],r.transform[1],r.transform[2],r.kernel.a,r.kernel.b,r.kernel.c,
            r.penalty,r.fit.transmission,r.fit.cost,r.fit.residual_cost,
            static_cast<double>(r.iterations),static_cast<double>(r.evaluations),r.converged?1.:0.};
        for(std::size_t i=0;i<13;++i) output[i]=values[i];
        return 0;
    } catch(const std::exception& e) {if(error && error_size) std::snprintf(error,error_size,"%s",e.what());return 1;}
    catch(...) {if(error && error_size) std::snprintf(error,error_size,"unknown C++ fit error");return 2;}
}
