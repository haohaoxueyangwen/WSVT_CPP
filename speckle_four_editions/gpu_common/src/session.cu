#include "speckle_native.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
int wxst_features_fused_gpu(const float*,int,int,int,float*,void*);
int edge_match(const float*,const float*,const float*,const float*,int,int,int,int,float*,float*,float*,void*);
int edge_sample_search(const float*,const float*,const float*,const float*,int,int,int,float*,float*,float*,void*);
int edge_spline_gpu(const float*,const float*,int,int,int,int,const double*,const double*,const int*,const double*,const double*,const double*,const int*,const double*,double*,double*,double*,float*,float*,double,void*);
int edge_stats100(const float*,int,double*,double*,void*);
int edge_stats100_u16(const unsigned short*,int,double*,double*,void*);
int edge_normalize100(const float*,int,const double*,const double*,float*,void*);
int edge_normalize100_u16(const unsigned short*,int,const double*,const double*,float*,void*);
int edge_temporal100(const float*,int,const double*,const double*,int,float*,void*);
int edge_temporal100_u16(const unsigned short*,int,const double*,const double*,int,float*,void*);
int edge_spatial(const float*,int,int,int,int,float*,float*,void*);
int edge_spatial_normalized100(const float*,int,int,const double*,const double*,int,float*,float*,void*);
int edge_spatial_normalized100_u16(const unsigned short*,int,int,const double*,const double*,int,float*,float*,void*);
int edge_copy_first_u16(const unsigned short*,int,float*,void*);
int edge_transmission(const float*,const float*,int,float*,void*);
int edge_transmission_u16(const unsigned short*,const float*,int,float*,void*);
int edge_post(const float*,const float*,const float*,int,int,int,double,double*,double*,float*,float*,float*,float*,float*,void*);
}
namespace {
thread_local std::string error_text;
constexpr double PI=3.1415926535897932384626433832795;
const char* outputs_names[]={"displace_y","displace_x","DPC_y","DPC_x","phase","transmission","match_score_neg_ssd"};
const char* descriptors[]={"ref_fine","ref_coarse","sample_fine","sample_coarse"};
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void fftck(cufftResult x){if(x!=CUFFT_SUCCESS)throw std::runtime_error("cuFFT status "+std::to_string(int(x)));}
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Buffer {
    void* p=nullptr;size_t n=0;
    Buffer()=default;Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(p)cudaFree(p);}
};
struct Stream {cudaStream_t s=nullptr;Stream(){ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
struct Event {cudaEvent_t e=nullptr;Event(){ck(cudaEventCreate(&e));}~Event(){if(e)cudaEventDestroy(e);}};
struct Plan {cufftHandle p=0;Plan(){fftck(cufftCreate(&p));fftck(cufftSetAutoAllocation(p,0));}~Plan(){if(p)cufftDestroy(p);}};
__global__ void limit_displacement(float* y,float* x,int n,float limit){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<n){if(!isnan(y[p]))y[p]=fminf(limit,fmaxf(-limit,y[p]));if(!isnan(x[p]))x[p]=fminf(limit,fmaxf(-limit,x[p]));}}
__global__ void make_frequencies(float* wx,float* wy,float* den,int h,int w,float sx,float sy){
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<w)wx[p]=float(p<=w/2?p:p-w)*sx;
    if(p<h)wy[p]=float(p<=h/2?p:p-h)*sy;
    if(p<h*w){int y=p/w,x=p%w;float fx=float(x<=w/2?x:x-w)*sx,fy=float(y<=h/2?y:y-h)*sy;den[p]=fmaxf(1e-12f,fy*fy+fx*fx);}
}
__device__ cufftComplex full_frequency(const cufftComplex* a,int y,int x,int h,int w){
    int sw=w/2+1;if(x<sw)return a[y*sw+x];
    cufftComplex v=a[((h-y)%h)*sw+w-x];v.y=-v.y;return v;
}
__global__ void integrate_spectrum(const cufftComplex* ax,const cufftComplex* ay,cufftComplex* out,const float* wx,const float* wy,const float* den,int h,int w){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=h*w)return;
    int y=p/w,x=p%w;cufftComplex a=full_frequency(ax,y,x,h,w),b=full_frequency(ay,y,x,h,w);
    // Same full-spectrum -i*(wx*Fx + wy*Fy)/den and positive Nyquist convention.
    float xr=wx[x]*a.y,xi=-wx[x]*a.x,yr=-wy[y]*b.y,yi=wy[y]*b.x;
    out[p].x=(xr-yr)/den[p];out[p].y=(xi-yi)/den[p];
}
__global__ void inverse_real(const cufftComplex* in,float* out,int n,float scale){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<n)out[p]=in[p].x*scale;}
__global__ void phase_partial_mean(const float* in,double* sums,int n){
    __shared__ double s[256];int p=blockIdx.x*256+threadIdx.x;s[threadIdx.x]=p<n?double(in[p]):0.;__syncthreads();
    for(int k=128;k;k/=2){if(threadIdx.x<k)s[threadIdx.x]+=s[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)sums[blockIdx.x]=s[0];
}
__global__ void phase_mean(const double* sums,float* mean,int blocks,int n){
    __shared__ double s[256];double v=0.;for(int j=threadIdx.x;j<blocks;j+=256)v+=sums[j];s[threadIdx.x]=v;__syncthreads();
    for(int k=128;k;k/=2){if(threadIdx.x<k)s[threadIdx.x]+=s[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)*mean=float(s[0]/double(n));
}
__global__ void finish_phase(float* out,const float* mean,int n,float scale){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<n)out[p]=(out[p]-*mean)*scale;}

class Session {
public:
    SpeckleConfig c;int h,w,hc,wc,oh,ow,depth,frames;size_t input_bytes=0,resident=0,fft_work_bytes=0;
    std::thread::id owner=std::this_thread::get_id();bool ready=false;double reference_time=0.;
    Stream stream;Plan forward,inverse;Event starts[8],ends[8];std::map<std::string,Buffer> b;
    const void* raw=nullptr;
    Session(SpeckleConfig config):c(config),h(c.height),w(c.width),hc((h+5)/2),wc((w+5)/2),oh(h-20),ow(w-20),depth(c.mode?63:58),frames(c.mode?100:1){
        if(h<=28||w<=28||int64_t(h)*w*100>2147483647LL||c.mode<0||c.mode>1||c.input_u16<0||c.input_u16>1||(!c.mode&&c.input_u16)||c.spatial_chunk<1||c.spatial_chunk>100)throw std::invalid_argument("Unsupported shape/mode/dtype/chunk; WXST float32, WSVT100 uint16/float32, half10/L2/I1/cut1");
        for(double v:{c.pixel_m,c.distance_m,c.energy_ev})if(!std::isfinite(v)||v<=0)throw std::invalid_argument("Explicit positive finite geometry required");
        cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,c.device));if(c.mapped_input&&!prop.integrated)throw std::invalid_argument("Mapped-input preset is restricted to integrated GPU; select copy on discrete GPU");
        const size_t p=size_t(h)*w,cp=size_t(hc)*wc,op=size_t(oh)*ow;input_bytes=p*frames*(c.input_u16?2:4);
        if(!c.mapped_input)alloc("raw",input_bytes);
        for(auto n:{"ref_first","transmission"})alloc(n,p*4);
        alloc("ref_fine",p*depth*4);alloc("ref_coarse",cp*depth*4);
        if(c.mode){
            alloc("sample_fine",p*depth*4);alloc("sample_coarse",cp*depth*4);alloc("mean",p*8);alloc("std",p*8);
            alloc("spatial_tmp",size_t(c.low_memory?c.spatial_chunk:100)*h*wc*4);alloc("coarse_stack",cp*100*4);
            if(!c.low_memory)alloc("normalized",p*100*4);
        }else{alloc("coarse_image",cp*4);alloc("spatial_tmp",size_t(h)*wc*4);}
        for(auto level:{"coarse","fine"})for(auto field:{"y","x","score"})alloc(std::string(level)+"_"+field,(std::string(level)=="coarse"?cp:p)*4);
        alloc("zero",cp*4);ck(cudaMemsetAsync(ptr("zero"),0,cp*4,stream.s));
        for(auto n:{"guess_y","guess_x"})alloc(n,p*4);
        for(auto n:{"displace_y","displace_x","DPC_y","DPC_x","phase","match_score_neg_ssd"})alloc(n,op*4);
        alloc("sums",2*((op+255)/256)*8);alloc("means",2*8);
        axis("sx",wc,w);axis("sy",hc,h);
        alloc("spline_rhs_x",size_t(wc-2)*2*hc*8);alloc("spline_rhs_y",size_t(hc-2)*2*w*8);alloc("spline_horizontal",size_t(2)*hc*w*8);
        alloc("wx",ow*4);alloc("wy",oh*4);alloc("den",op*4);
        alloc("rfft_x",size_t(oh)*(ow/2+1)*sizeof(cufftComplex));alloc("rfft_y",size_t(oh)*(ow/2+1)*sizeof(cufftComplex));
        alloc("spectrum",op*sizeof(cufftComplex));alloc("ifft",op*sizeof(cufftComplex));
        alloc("phase_sums",((op+255)/256)*8);alloc("phase_mean",4);
        size_t fw=0,iw=0;fftck(cufftMakePlan2d(forward.p,oh,ow,CUFFT_R2C,&fw));fftck(cufftMakePlan2d(inverse.p,oh,ow,CUFFT_C2C,&iw));
        fft_work_bytes=std::max(fw,iw);alloc("fft_workspace",fft_work_bytes);
        fftck(cufftSetWorkArea(forward.p,ptr("fft_workspace")));fftck(cufftSetWorkArea(inverse.p,ptr("fft_workspace")));
        fftck(cufftSetStream(forward.p,stream.s));fftck(cufftSetStream(inverse.p,stream.s));
        make_frequencies<<<(op+255)/256,256,0,stream.s>>>(f("wx"),f("wy"),f("den"),oh,ow,float(2*PI/ow),float(2*PI/oh));ck(cudaGetLastError());ck(cudaStreamSynchronize(stream.s));
    }
    ~Session(){cudaSetDevice(c.device);cudaStreamSynchronize(stream.s);}
    void alloc(const std::string& n,size_t bytes){auto& v=b[n];v.n=bytes;if(bytes)ck(cudaMalloc(&v.p,bytes));resident+=bytes;}
    void* ptr(const std::string& n){return b.at(n).p;}
    float* f(const std::string& n){return static_cast<float*>(ptr(n));}
    double* d(const std::string& n){return static_cast<double*>(ptr(n));}
    cufftComplex* z(const std::string& n){return static_cast<cufftComplex*>(ptr(n));}
    void check_owner(){if(owner!=std::this_thread::get_id())throw std::runtime_error("Session belongs to its creating thread");ck(cudaSetDevice(c.device));}
    void validate(const void* a,size_t bytes){check_owner();if(!a||bytes!=input_bytes)throw std::invalid_argument("Input byte count/shape differs from session");if(!c.input_u16){const float* p=static_cast<const float*>(a);for(size_t j=0;j<bytes/4;++j)if(!std::isfinite(p[j]))throw std::invalid_argument("Nonfinite input");}}
    void upload(const void* a){if(c.mapped_input){void* p=nullptr;ck(cudaHostGetDevicePointer(&p,const_cast<void*>(a),0));raw=p;}else{raw=ptr("raw");ck(cudaMemcpyAsync(ptr("raw"),a,input_bytes,cudaMemcpyHostToDevice,stream.s));}}
    void axis(const std::string& name,int n,int m){
        std::vector<double> diag(n-2,4.),upper(n-2,0.),factor(n-2,0.),t(m);std::vector<int> left(m);diag.front()=diag.back()=1.;
        for(int j=1;j<n-3;++j)upper[j]=1.;
        for(int j=1;j<n-2;++j){factor[j]=(j==n-3?0.:1.)/diag[j-1];diag[j]-=factor[j]*upper[j-1];}
        for(int j=0;j<m;++j){double x=double(j)*(double(n-1)/double(m-1));left[j]=std::min(int(std::floor(x)),n-2);t[j]=x-left[j];}
        auto put=[&](const char* suffix,const void* a,size_t bytes){std::string key=name+"_"+suffix;alloc(key,bytes);ck(cudaMemcpy(ptr(key),a,bytes,cudaMemcpyHostToDevice));};
        put("diag",diag.data(),diag.size()*8);put("factor",factor.data(),factor.size()*8);put("left",left.data(),left.size()*4);put("t",t.data(),t.size()*8);
    }
    void spatial(const float* in,int channels,float* out){ck(cudaError_t(edge_spatial(in,h,w,channels,0,f("spatial_tmp"),out,stream.s)));}
    void features(const char* fine,const char* coarse){
        int p=h*w;const auto ru=static_cast<const unsigned short*>(raw);const auto rf=static_cast<const float*>(raw);
        if(!c.mode){spatial(rf,1,f("coarse_image"));ck(cudaError_t(wxst_features_fused_gpu(rf,h,w,2,f(fine),stream.s)));ck(cudaError_t(wxst_features_fused_gpu(f("coarse_image"),hc,wc,2,f(coarse),stream.s)));return;}
        ck(cudaError_t(c.input_u16?edge_stats100_u16(ru,p,d("mean"),d("std"),stream.s):edge_stats100(rf,p,d("mean"),d("std"),stream.s)));
        if(c.low_memory){
            ck(cudaError_t(c.input_u16?edge_spatial_normalized100_u16(ru,h,w,d("mean"),d("std"),c.spatial_chunk,f("spatial_tmp"),f("coarse_stack"),stream.s):edge_spatial_normalized100(rf,h,w,d("mean"),d("std"),c.spatial_chunk,f("spatial_tmp"),f("coarse_stack"),stream.s)));
            ck(cudaError_t(c.input_u16?edge_temporal100_u16(ru,p,d("mean"),d("std"),1,f(fine),stream.s):edge_temporal100(rf,p,d("mean"),d("std"),1,f(fine),stream.s)));
        }else{
            ck(cudaError_t(c.input_u16?edge_normalize100_u16(ru,p,d("mean"),d("std"),f("normalized"),stream.s):edge_normalize100(rf,p,d("mean"),d("std"),f("normalized"),stream.s)));
            spatial(f("normalized"),100,f("coarse_stack"));ck(cudaError_t(edge_temporal100(f("normalized"),p,d("mean"),d("std"),0,f(fine),stream.s)));
        }
        ck(cudaError_t(edge_temporal100(f("coarse_stack"),hc*wc,d("mean"),d("std"),0,f(coarse),stream.s)));
    }
    double prepare(const void* a,size_t bytes){validate(a,bytes);ready=false;double t=now();upload(a);
        if(c.input_u16)ck(cudaError_t(edge_copy_first_u16(static_cast<const unsigned short*>(raw),h*w,f("ref_first"),stream.s)));
        else ck(cudaMemcpyAsync(f("ref_first"),raw,size_t(h)*w*4,cudaMemcpyDeviceToDevice,stream.s));
        features("ref_fine","ref_coarse");ck(cudaStreamSynchronize(stream.s));ready=true;reference_time=now()-t;return reference_time;
    }
    void match(bool coarse){
        int hh=coarse?hc:h,ww=coarse?wc:w,r=coarse?5:4;std::string k=coarse?"coarse":"fine";
        const float* iy=f(coarse?"zero":"guess_y"),*ix=f(coarse?"zero":"guess_x");
        if(c.mode)ck(cudaError_t(edge_match(f("sample_"+k),f("ref_"+k),iy,ix,hh,ww,depth,r,f(k+"_y"),f(k+"_x"),f(k+"_score"),stream.s)));
        else ck(cudaError_t(edge_sample_search(coarse?f("coarse_image"):static_cast<const float*>(raw),f("ref_"+k),iy,ix,hh,ww,r,f(k+"_y"),f(k+"_x"),f(k+"_score"),stream.s)));
        limit_displacement<<<(hh*ww+255)/256,256,0,stream.s>>>(f(k+"_y"),f(k+"_x"),hh*ww,coarse?5.f:10.f);ck(cudaGetLastError());
    }
    void spline(){ck(cudaError_t(edge_spline_gpu(f("coarse_y"),f("coarse_x"),hc,wc,h,w,d("sx_diag"),d("sx_factor"),static_cast<int*>(ptr("sx_left")),d("sx_t"),d("sy_diag"),d("sy_factor"),static_cast<int*>(ptr("sy_left")),d("sy_t"),d("spline_rhs_x"),d("spline_rhs_y"),d("spline_horizontal"),f("guess_y"),f("guess_x"),10.,stream.s)));}
    void phase(){
        int n=oh*ow,blocks=(n+255)/256;
        fftck(cufftExecR2C(forward.p,f("DPC_x"),z("rfft_x")));fftck(cufftExecR2C(forward.p,f("DPC_y"),z("rfft_y")));
        integrate_spectrum<<<blocks,256,0,stream.s>>>(z("rfft_x"),z("rfft_y"),z("spectrum"),f("wx"),f("wy"),f("den"),oh,ow);ck(cudaGetLastError());
        fftck(cufftExecC2C(inverse.p,z("spectrum"),z("ifft"),CUFFT_INVERSE));
        inverse_real<<<blocks,256,0,stream.s>>>(z("ifft"),f("phase"),n,float(1./double(n)));
        phase_partial_mean<<<blocks,256,0,stream.s>>>(f("phase"),d("phase_sums"),n);phase_mean<<<1,256,0,stream.s>>>(d("phase_sums"),f("phase_mean"),blocks,n);
        finish_phase<<<blocks,256,0,stream.s>>>(f("phase"),f("phase_mean"),n,float(-c.pixel_m*2*PI/(1.2398419843320026e-6/c.energy_ev)));ck(cudaGetLastError());
    }
    template<class F>void stage(int k,F fn){ck(cudaEventRecord(starts[k].e,stream.s));fn();ck(cudaEventRecord(ends[k].e,stream.s));}
    SpeckleTiming solve(const void* a,size_t bytes,void** outputs){
        double call=now();validate(a,bytes);if(!ready)throw std::runtime_error("Prepare reference before sample solve");
        if(outputs)for(int j=0;j<7;++j)if(!outputs[j])throw std::invalid_argument("Provide all seven writable output buffers");
        double t=now();stage(0,[&]{upload(a);});
        stage(1,[&]{ck(cudaError_t(c.input_u16?edge_transmission_u16(static_cast<const unsigned short*>(raw),f("ref_first"),h*w,f("transmission"),stream.s):edge_transmission(static_cast<const float*>(raw),f("ref_first"),h*w,f("transmission"),stream.s)));});
        stage(2,[&]{if(c.mode)features("sample_fine","sample_coarse");else spatial(static_cast<const float*>(raw),1,f("coarse_image"));});
        stage(3,[&]{match(true);});stage(4,[&]{spline();});stage(5,[&]{match(false);});
        stage(6,[&]{ck(cudaError_t(edge_post(f("fine_y"),f("fine_x"),f("fine_score"),h,w,10,c.pixel_m/c.distance_m,d("sums"),d("means"),f("displace_y"),f("displace_x"),f("DPC_y"),f("DPC_x"),f("match_score_neg_ssd"),stream.s)));});stage(7,[&]{phase();});
        ck(cudaStreamSynchronize(stream.s));SpeckleTiming info{};info.reference_s=reference_time;info.sample_solver_s=now()-t;t=now();
        if(outputs){for(int j=0;j<7;++j){const auto& v=b.at(outputs_names[j]);ck(cudaMemcpyAsync(outputs[j],v.p,v.n,cudaMemcpyDeviceToHost,stream.s));}ck(cudaStreamSynchronize(stream.s));}
        info.download_s=now()-t;info.call_s=now()-call;info.persistent_bytes=resident-fft_work_bytes;info.fft_workspace_bytes=fft_work_bytes;info.device_allocation_bytes=resident;
        for(int j=0;j<8;++j){float ms=0;ck(cudaEventElapsedTime(&ms,starts[j].e,ends[j].e));info.stages_s[j]=ms/1000.;}return info;
    }
};
template<class F>int guard(F fn){try{fn();error_text.clear();return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
}
extern "C" const char* speckle_error(){return error_text.c_str();}
extern "C" void* speckle_create(const SpeckleConfig* c){try{if(!c)throw std::invalid_argument("Null config");ck(cudaSetDevice(c->device));auto* p=new Session(*c);error_text.clear();return p;}catch(const std::exception& e){error_text=e.what();return nullptr;}}
extern "C" void speckle_destroy(void* p){if(p)delete static_cast<Session*>(p);}
extern "C" int speckle_prepare(void* p,const void* a,size_t bytes,double* seconds){return guard([&]{if(!p||!seconds)throw std::invalid_argument("Null handle/timing");*seconds=static_cast<Session*>(p)->prepare(a,bytes);});}
extern "C" int speckle_solve(void* p,const void* a,size_t bytes,void** outputs,SpeckleTiming* info){return guard([&]{if(!p||!info)throw std::invalid_argument("Null handle/timing");*info=static_cast<Session*>(p)->solve(a,bytes,outputs);});}
extern "C" uint64_t speckle_descriptor_bytes(void* p,int index){if(!p||index<0||index>3)return 0;auto* q=static_cast<Session*>(p);auto i=q->b.find(descriptors[index]);return i==q->b.end()?0:i->second.n;}
extern "C" int speckle_copy_descriptor(void* p,int index,void* out,size_t bytes){return guard([&]{if(!p||!out||index<0||index>3||bytes!=speckle_descriptor_bytes(p,index))throw std::invalid_argument("Invalid descriptor buffer/index");auto* q=static_cast<Session*>(p);q->check_owner();ck(cudaMemcpy(out,q->ptr(descriptors[index]),bytes,cudaMemcpyDeviceToHost));});}
extern "C" void* speckle_host_alloc(size_t bytes,int mapped){void* p=nullptr;try{ck(cudaHostAlloc(&p,bytes,mapped?cudaHostAllocMapped:cudaHostAllocDefault));return p;}catch(const std::exception& e){error_text=e.what();return nullptr;}}
extern "C" int speckle_host_free(void* p){return guard([&]{if(p)ck(cudaFreeHost(p));});}
extern "C" int speckle_device_info(int device,char* out,size_t capacity){return guard([&]{cudaDeviceProp p;ck(cudaGetDeviceProperties(&p,device));std::ostringstream s;s<<p.name<<"; SM"<<p.major<<p.minor<<"; integrated="<<p.integrated<<"; canMapHostMemory="<<p.canMapHostMemory;auto v=s.str();if(!out||capacity<=v.size())throw std::invalid_argument("Device info buffer too small");std::memcpy(out,v.c_str(),v.size()+1);});}
