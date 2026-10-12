// Independent fixed-grid CPU spline plan; existing WSVT_CPP stays untouched.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
namespace {
thread_local std::string error;
struct Axis {
    int n, m; std::vector<double> diag, factor, upper, t; std::vector<int> left;
    Axis(int input, int output):n(input),m(output),diag(n-2,4.),factor(n-2,0.),upper(n-2,0.),t(m),left(m) {
        if(n<4||m<2)throw std::invalid_argument("spline dimensions require input>=4/output>=2");
        diag.front()=diag.back()=1.;
        for(int j=1;j<n-3;++j)upper[j]=1.;
        for(int j=1;j<n-2;++j){factor[j]=(j==n-3?0.:1.)/diag[j-1];diag[j]-=factor[j]*upper[j-1];}
        const double scale=double(n-1)/double(m-1);
        for(int j=0;j<m;++j){double q=scale*j;left[j]=std::min(n-2,int(std::floor(q)));t[j]=q-left[j];}
    }
    void second(const double* v,int stride,double* rhs,double* out)const {
        for(int j=0;j<n-2;++j){int i=j+1;rhs[j]=(v[(i+1)*stride]-2.*v[i*stride]+v[(i-1)*stride])*(j==0||j==n-3?1.:6.);}
        for(int j=1;j<n-2;++j)rhs[j]-=factor[j]*rhs[j-1];
        rhs[n-3]/=diag[n-3];
        for(int j=n-4;j>=0;--j)rhs[j]=(rhs[j]-upper[j]*rhs[j+1])/diag[j];
        for(int j=1;j<n-1;++j)out[j]=rhs[j-1];
        out[0]=2.*out[1]-out[2];out[n-1]=2.*out[n-2]-out[n-3];
    }
    double eval(const double* v,int stride,const double* second,int j)const {
        const int l=left[j];const double u=t[j],a=1.-u;
        return second[l]*a*a*a/6.+second[l+1]*u*u*u/6.+(v[l*stride]-second[l]/6.)*a+(v[(l+1)*stride]-second[l+1]/6.)*u;
    }
};
struct Plan {
    int h,w,oh,ow,threads;Axis x,y;
    std::vector<double> scaled,horizontal,values,rhs,second;
    Plan(int a,int b,int c,int d,int nt):h(a),w(b),oh(c),ow(d),threads(std::max(1,nt)),x(b,d),y(a,c),scaled(2ull*a*b),horizontal(2ull*a*d),values(1ull*threads*std::max(a,b)),rhs(values.size()),second(values.size()){}
    void run(const float* iy,const float* ix,float* oy,float* ox,double limit) {
        // v3 convention: float scaling before spline; two components together.
        for(std::size_t p=0;p<std::size_t(h)*w;++p){scaled[p]=double(iy[p]*2.f);scaled[std::size_t(h)*w+p]=double(ix[p]*2.f);}
        #pragma omp parallel num_threads(threads)
        {
            int tid=0;
            #ifdef _OPENMP
            tid=omp_get_thread_num();
            #endif
            int stride=std::max(h,w);double* v=values.data()+std::size_t(tid)*stride;
            double* r=rhs.data()+std::size_t(tid)*stride;double* s=second.data()+std::size_t(tid)*stride;
            #pragma omp for schedule(static)
            for(int line=0;line<2*h;++line){const double* row=scaled.data()+std::size_t(line)*w;x.second(row,1,r,s);
                for(int j=0;j<ow;++j)horizontal[std::size_t(line)*ow+j]=x.eval(row,1,s,j);}
            #pragma omp for schedule(static)
            for(int line=0;line<2*ow;++line){const int component=line/ow,col=line%ow;
                for(int i=0;i<h;++i)v[i]=horizontal[(std::size_t(component)*h+i)*ow+col];
                y.second(v,1,r,s);float* output=component?ox:oy;
                for(int j=0;j<oh;++j){double value=std::nearbyint(y.eval(v,1,s,j));output[std::size_t(j)*ow+col]=float(std::max(-limit,std::min(limit,value)));}}
        }
    }
};
}
API const char* edge_spline_error(){return error.c_str();}
API void* edge_spline_create(int h,int w,int oh,int ow,int threads){try{if(h<4||w<4||oh<2||ow<2||threads<1)throw std::invalid_argument("invalid spline plan");return new Plan(h,w,oh,ow,threads);}catch(const std::exception& e){error=e.what();return nullptr;}}
API int edge_spline_run(void* handle,const float* iy,const float* ix,float* oy,float* ox,double limit){
    try{if(!handle||!iy||!ix||!oy||!ox||!std::isfinite(limit)||limit<0)throw std::invalid_argument("invalid spline input");static_cast<Plan*>(handle)->run(iy,ix,oy,ox,limit);return 0;}
    catch(const std::exception& e){error=e.what();return 1;}}
API void edge_spline_destroy(void* handle){delete static_cast<Plan*>(handle);}
