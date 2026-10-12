#include "wsvt/wsvt_pipeline.hpp"
#include "wsvt/wxst_pipeline.hpp"
#include <chrono>
#include <exception>
#include <string>
#include <vector>
#include <utility>

struct Result {
    std::vector<float> fields[7];
    size_t h, w, th, tw;
    double times[10]{};
};
static thread_local std::string error;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point a) {return std::chrono::duration<double>(Clock::now()-a).count();}
template<class T> static void assign(Result& r,T&& o) {
    r.h=o.h;r.w=o.w;r.th=o.transmission_h;r.tw=o.transmission_w;
    r.fields[0]=std::move(o.displace_y);r.fields[1]=std::move(o.displace_x);
    r.fields[2]=std::move(o.dpc_y);r.fields[3]=std::move(o.dpc_x);
    r.fields[4]=std::move(o.phase);r.fields[5]=std::move(o.transmission);
    r.fields[6]=std::move(o.darkfield_nd);
    r.times[3]=o.time_cost_s;r.times[4]=o.pyramid_time_s;r.times[5]=o.wavelet_time_s;
    r.times[6]=o.displace_time_s;r.times[7]=o.postprocess_time_s;
    r.times[8]=o.post_phase_recovery_time_s;
}
extern "C" {
const char* cpu_error(){return error.c_str();}
void* cpu_solve(int mode,const float* sample,const float* reference,size_t n,size_t h,size_t w,int threads,int phase_threads) {
    Result* result=nullptr;
    try {
        auto start=Clock::now();const size_t count=n*h*w;
        if(!sample || !reference || n==0 || h==0 || w==0) throw std::invalid_argument("Invalid input");
        std::span<const float> s(sample,count),r(reference,count);
        double copy_time=elapsed(start);result=new Result();
        auto ctor=Clock::now();
        if(mode==0) {
            wsvt::WSVT solver(s,r,n,h,w,0,10,0,4,threads,4,12400.,1.625e-6,1.,.4845,1,1,1,false,true,0,false,phase_threads);
            result->times[1]=elapsed(ctor);auto solve_start=Clock::now();
            auto o=solver.solver();result->times[2]=elapsed(solve_start);assign(*result,std::move(o));
        } else {
            wsvt::WXST solver(s,r,h,w,int(h),3,10,4,threads,4,12400.,1.625e-6,.4845,1,1,1,false,true,0,2,phase_threads);
            result->times[1]=elapsed(ctor);auto solve_start=Clock::now();
            auto o=solver.solver();result->times[2]=elapsed(solve_start);assign(*result,std::move(o));
        }
        result->times[0]=copy_time;result->times[9]=elapsed(start);return result;
    }catch(const std::exception& e){error=e.what();delete result;return nullptr;}
}
const float* cpu_field(void* handle,int field,size_t* h,size_t* w) {
    auto& r=*static_cast<Result*>(handle);*h=field==5?r.th:r.h;*w=field==5?r.tw:r.w;
    return r.fields[field].data();
}
const double* cpu_times(void* handle){return static_cast<Result*>(handle)->times;}
void cpu_free(void* handle){delete static_cast<Result*>(handle);}
}
