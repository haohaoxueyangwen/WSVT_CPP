// New isolated edge backend. Original v1/v2/v3 sources are only included, never edited.
#include "cached_match.cuh"

namespace edge {
constexpr int WARPS=4, COSTS=121;
__device__ void finish(float* costs,int side,int winner,float best,float gy0,float gx0,
                       int radius,int p,float* oy,float* ox,float* score){
    const int by=winner/side,bx=winner%side;float patch[9];
    #pragma unroll
    for(int py=-1;py<=1;++py){
        #pragma unroll
        for(int px=-1;px<=1;++px)patch[(py+1)*3+px+1]=-costs[max(0,min(side-1,by+py))*side+max(0,min(side-1,bx+px))];}
    float gy=(patch[7]-patch[1])*.5f,gx=(patch[5]-patch[3])*.5f;
    float dyy=patch[7]+patch[1]-2.f*patch[4],dxx=patch[5]+patch[3]-2.f*patch[4];
    float dxy=(patch[8]-patch[6]-patch[2]+patch[0])*.25f;
    float det=dxx*dyy-dxy*dxy,inv=det!=0.f?1.f/det:0.f;
    float dx=-(dyy*gx-dxy*gy)*inv,dy=-(dxx*gy-dxy*gx)*inv;
    ox[p]=fminf(radius,fmaxf(-radius,bx-radius+dx))+gx0;
    oy[p]=fminf(radius,fmaxf(-radius,by-radius+dy))+gy0;score[p]=-best;
}
__device__ void search(float a0,float a1,float a2,float a3,const float* ref,
                      int h,int w,int d,int radius,int p,float gy,float gx,
                      float* costs,float* oy,float* ox,float* score){
    int lane=threadIdx.x&31,side=2*radius+1,y0=p/w+int(gy),x0=p%w+int(gx),winner=0;
    float best=CUDART_INF_F;
    for(int cy=0;cy<side;++cy)for(int cx=0;cx<side;++cx){
        float cost=cached_cost(a0,a1,a2,a3,ref,y0+cy-radius,x0+cx-radius,h,w,d);
        if(lane==0){int j=cy*side+cx;costs[j]=cost;if(cost<best){best=cost;winner=j;}}}
    if(lane==0)finish(costs,side,winner,best,gy,gx,radius,p,oy,ox,score);
}
__global__ void match_costs(const float* a,const float* b,const float* iy,const float* ix,
                            int h,int w,int d,int radius,float* oy,float* ox,float* score){
    int lane=threadIdx.x&31,warp=threadIdx.x/32,p=blockIdx.x*WARPS+warp;if(p>=h*w)return;
    __shared__ float costs[WARPS][COSTS];std::size_t q=std::size_t(p)*d;
    search(lane<d?a[q+lane]:0.f,lane+32<d?a[q+lane+32]:0.f,
           lane+64<d?a[q+lane+64]:0.f,lane+96<d?a[q+lane+96]:0.f,
           b,h,w,d,radius,p,iy[p],ix[p],costs[warp],oy,ox,score);
}
__global__ void sample_search(const float* image,const float* ref,const float* iy,const float* ix,
                              int h,int w,int radius,float* oy,float* ox,float* score){
    int lane=threadIdx.x&31,warp=threadIdx.x/32,p=blockIdx.x*WARPS+warp;if(p>=h*w)return;
    __shared__ float a[WARPS][64],b[WARPS][64],f[WARPS][64],costs[WARPS][COSTS];
    for(int k=lane;k<49;k+=32){int dy=k/7-3,dx=k%7-3;a[warp][k]=image[((p/w-dy+h)%h)*w+(p%w-dx+w)%w];}
    __syncwarp(MASK);float mean=0.f,inv=0.f;
    if(lane==0){float sum=0.f,sq=0.f;for(int k=0;k<49;++k){float v=a[warp][k];sum+=v;sq+=v*v;}
        mean=sum*(1.f/49.f);inv=1.f/(sqrtf(fmaxf(0.f,sq*(1.f/49.f)-mean*mean))+1e-6f);}
    mean=__shfl_sync(MASK,mean,0);inv=__shfl_sync(MASK,inv,0);
    for(int k=lane;k<49;k+=32)a[warp][k]=(a[warp][k]-mean)*inv;
    __syncwarp(MASK);int lengths[4]={26,14,8,5},offsets[4]={32,18,10,5},d=49;
    float* cur=a[warp];float* next=b[warp];
    for(int lv=0;lv<4;++lv){if(lane<lengths[lv]){float ca=0.f,cd=0.f;
        for(int j=0;j<4;++j){int ic=2*lane+j-2;if(ic>=0&&ic<d){ca+=db2_lo[j]*cur[ic];cd+=db2_hi[j]*cur[ic];}}
        next[lane]=ca;f[warp][offsets[lv]+lane]=cd;}
        __syncwarp(MASK);float* tmp=cur;cur=next;next=tmp;d=lengths[lv];}
    if(lane<5)f[warp][lane]=cur[lane];__syncwarp(MASK);
    search(f[warp][lane],lane<26?f[warp][lane+32]:0.f,0.f,0.f,ref,h,w,58,radius,p,
           iy[p],ix[p],costs[warp],oy,ox,score);
}

// Fixed-grid not-a-knot spline. RHS and solved second derivatives are depth-major,
// so different line systems read adjacent addresses while each line recurses.
__global__ void spline_rhs_rows(const float* iy,const float* ix,int h,int w,double* rhs){
    int p=blockIdx.x*blockDim.x+threadIdx.x,systems=2*h,count=w-2;if(p>=systems*count)return;
    int j=p/systems,s=p%systems,c=s/h,y=s%h;const float* v=(c?ix:iy)+std::size_t(y)*w;
    rhs[p]=(double(v[j+2]*2.f)-2.*double(v[j+1]*2.f)+double(v[j]*2.f))*(j==0||j==count-1?1.:6.);
}
__global__ void spline_solve(double* rhs,int systems,int count,const double* diag,const double* factor){
    int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=systems)return;
    for(int j=1;j<count;++j)rhs[std::size_t(j)*systems+s]-=factor[j]*rhs[std::size_t(j-1)*systems+s];
    rhs[std::size_t(count-1)*systems+s]/=diag[count-1];
    for(int j=count-2;j>=0;--j)rhs[std::size_t(j)*systems+s]=(rhs[std::size_t(j)*systems+s]-(j>0&&j<count-1?rhs[std::size_t(j+1)*systems+s]:0.))/diag[j];
}
__device__ double second(const double* rhs,int s,int systems,int n,int i){
    if(i==0)return 2.*rhs[s]-rhs[systems+s];
    if(i==n-1)return 2.*rhs[std::size_t(n-3)*systems+s]-rhs[std::size_t(n-4)*systems+s];
    return rhs[std::size_t(i-1)*systems+s];
}
__device__ double eval(double v0,double v1,double m0,double m1,double t){
    double a=1.-t;return m0*a*a*a/6.+m1*t*t*t/6.+(v0-m0/6.)*a+(v1-m1/6.)*t;
}
__global__ void spline_rows_eval(const float* iy,const float* ix,int h,int w,int ow,
                                 const int* left,const double* t,const double* rhs,double* horizontal){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=2*h*ow)return;
    int s=p/ow,x=p%ow,c=s/h,y=s%h,l=left[x];const float* v=(c?ix:iy)+std::size_t(y)*w;
    horizontal[p]=eval(double(v[l]*2.f),double(v[l+1]*2.f),second(rhs,s,2*h,w,l),second(rhs,s,2*h,w,l+1),t[x]);
}
__global__ void spline_rhs_cols(const double* horizontal,int h,int ow,double* rhs){
    int p=blockIdx.x*blockDim.x+threadIdx.x,systems=2*ow,count=h-2;if(p>=systems*count)return;
    int j=p/systems,s=p%systems,c=s/ow,x=s%ow;const double* v=horizontal+std::size_t(c)*h*ow+x;
    rhs[p]=(v[std::size_t(j+2)*ow]-2.*v[std::size_t(j+1)*ow]+v[std::size_t(j)*ow])*(j==0||j==count-1?1.:6.);
}
__global__ void spline_cols_eval(const double* horizontal,int h,int oh,int ow,const int* left,
                                 const double* t,const double* rhs,float* oy,float* ox,double limit){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=2*oh*ow)return;
    int c=p/(oh*ow),y=p/ow%oh,x=p%ow,l=left[y],s=c*ow+x;
    const double* v=horizontal+std::size_t(c)*h*ow+x;
    double value=eval(v[std::size_t(l)*ow],v[std::size_t(l+1)*ow],second(rhs,s,2*ow,h,l),second(rhs,s,2*ow,h,l+1),t[y]);
    (c?ox:oy)[std::size_t(y)*ow+x]=float(fmin(limit,fmax(-limit,nearbyint(value))));
}

template<class T> __global__ void stats100(const T* raw,int pixels,double* mean,double* stddev){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=pixels)return;double sum=0.;
    for(int c=0;c<100;++c)sum+=double(raw[std::size_t(c)*pixels+p]);double mu=sum/100.,sq=0.;
    for(int c=0;c<100;++c){double v=double(raw[std::size_t(c)*pixels+p])-mu;sq+=v*v;}
    mean[p]=mu;stddev[p]=sqrt(sq/100.);
}
template<class T> __global__ void normalize100(const T* raw,int pixels,const double* mean,const double* sd,float* normalized){
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=100*pixels)return;int p=q%pixels;
    normalized[q]=sd[p]>0.?float((double(raw[q])-mean[p])/sd[p]):0.f;
}
template<class T> __global__ void temporal100(const T* raw,int pixels,const double* mean,const double* sd,int normalize,float* out){
    // Load spatially adjacent pixels coalesced, then use padding to avoid bank conflicts.
    __shared__ float tile[100][33],a[WARPS][128],b[WARPS][128];
    int tid=threadIdx.x,lane=tid&31,warp=tid/32,base=blockIdx.x*32;
    for(int q=tid;q<3200;q+=128){int c=q/32,x=q%32,p=base+x;float v=p<pixels?raw[std::size_t(c)*pixels+p]:0.f;
        if(normalize&&p<pixels)v=sd[p]>0.?float((double(v)-mean[p])/sd[p]):0.f;tile[c][x]=v;}
    __syncthreads();
    for(int x=warp;x<32;x+=WARPS){int p=base+x;if(p>=pixels)continue;
        for(int k=lane;k<100;k+=32)a[warp][k]=tile[k][x];__syncwarp(MASK);
        float* cur=a[warp];float* next=b[warp];int d=100,lengths[5]={51,27,15,9,6},offset[5]={-1,36,21,12,6};
        for(int lv=0;lv<5;++lv){for(int oc=lane;oc<lengths[lv];oc+=32){float ca=0.f,cd=0.f;
            for(int j=0;j<4;++j){int ic=2*oc+j-2;if(ic>=0&&ic<d){ca+=db2_lo[j]*cur[ic];if(lv>0)cd+=db2_hi[j]*cur[ic];}}
            next[oc]=ca;if(lv>0)out[std::size_t(p)*63+offset[lv]+oc]=cd;}
            __syncwarp(MASK);float* tmp=cur;cur=next;next=tmp;d=lengths[lv];}
        if(lane<6)out[std::size_t(p)*63+lane]=cur[lane];__syncwarp(MASK);
    }
}
__global__ void spatial_x(const float* in,int h,int w,int ow,int channels,int hwd,float* tmp){
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=channels*h*ow)return;
    int c,y,x;if(hwd){c=q%channels;x=q/channels%ow;y=q/(channels*ow);}else{x=q%ow;y=q/ow%h;c=q/(h*ow);}
    float sum=0.f;for(int j=0;j<6;++j){int ix=2*x+j-4;if(ix>=0&&ix<w){std::size_t p=hwd?(std::size_t(y)*w+ix)*channels+c:(std::size_t(c)*h+y)*w+ix;sum+=db3_reversed[j]*in[p];}}
    tmp[q]=sum;
}
__global__ void spatial_y(const float* tmp,int h,int oh,int ow,int channels,int hwd,float* out){
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=channels*oh*ow)return;
    int c,y,x;if(hwd){c=q%channels;x=q/channels%ow;y=q/(channels*ow);}else{x=q%ow;y=q/ow%oh;c=q/(oh*ow);}
    float sum=0.f;for(int j=0;j<6;++j){int iy=2*y+j-4;if(iy>=0&&iy<h){std::size_t p=hwd?(std::size_t(iy)*ow+x)*channels+c:(std::size_t(c)*h+iy)*ow+x;sum+=db3_reversed[j]*tmp[p];}}
    out[q]=sum;
}
// Normalize once per tile, keeping the old double calculation and float rounding.
// Neighboring outputs reuse the tile; invalid taps are skipped just as in spatial_x.
template<class T> __global__ void spatial_normalized_x(const T* raw,int h,int w,int ow,
    const double* mean,const double* sd,int c0,float* tmp){
    __shared__ float tile[260];
    int tid=threadIdx.x,y=blockIdx.y,c=blockIdx.z,x0=blockIdx.x*128,base=2*x0-4;
    for(int k=tid;k<260;k+=128){int ix=base+k;float v=0.f;
        if(ix>=0&&ix<w){std::size_t p=std::size_t(y)*w+ix;
            v=sd[p]>0.?float((double(raw[std::size_t(c0+c)*h*w+p])-mean[p])/sd[p]):0.f;}
        tile[k]=v;}
    __syncthreads();int x=x0+tid;if(x>=ow)return;
    float sum=0.f;for(int j=0;j<6;++j){int ix=2*x+j-4;
        if(ix>=0&&ix<w)sum+=db3_reversed[j]*tile[2*tid+j];}
    tmp[(std::size_t(c)*h+y)*ow+x]=sum;
}
template<class T> int spatial_normalized100(const T* raw,int h,int w,const double* mean,
    const double* sd,int chunk,float* tmp,float* out,void* stream){
    if(!raw||!mean||!sd||!tmp||!out||h<1||w<1||chunk<1||chunk>100)return cudaErrorInvalidValue;
    int oh=(h+5)/2,ow=(w+5)/2;auto s=reinterpret_cast<cudaStream_t>(stream);
    for(int c0=0;c0<100;c0+=chunk){int active=min(chunk,100-c0);
        spatial_normalized_x<<<dim3((ow+127)/128,h,active),128,0,s>>>(raw,h,w,ow,mean,sd,c0,tmp);
        cudaError_t code=cudaGetLastError();if(code!=cudaSuccess)return code;
        spatial_y<<<(active*oh*ow+255)/256,256,0,s>>>(tmp,h,oh,ow,active,0,out+std::size_t(c0)*oh*ow);
        code=cudaGetLastError();if(code!=cudaSuccess)return code;}
    return cudaSuccess;
}
template<class T> __global__ void transmission(const T* sample,const float* ref,int count,float* out){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count){double v=double(sample[p])/fmax(double(ref[p]),1e-10);out[p]=float(fmin(10.,fmax(.01,v)));}
}
__global__ void copy_first_u16(const unsigned short* raw,int count,float* out){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count)out[p]=float(raw[p]);
}
__global__ void crop_sums(const float* iy,const float* ix,int h,int w,int half,double* sums){
    int oh=h-2*half,ow=w-2*half,p=blockIdx.x*blockDim.x+threadIdx.x;double y=0.,x=0.;
    if(p<oh*ow){int q=(p/ow+half)*w+p%ow+half;y=-double(iy[q]);x=-double(ix[q]);}
    __shared__ double sy[256],sx[256];sy[threadIdx.x]=y;sx[threadIdx.x]=x;__syncthreads();
    for(int s=128;s;s>>=1){if(threadIdx.x<s){sy[threadIdx.x]+=sy[threadIdx.x+s];sx[threadIdx.x]+=sx[threadIdx.x+s];}__syncthreads();}
    if(threadIdx.x==0){sums[2*blockIdx.x]=sy[0];sums[2*blockIdx.x+1]=sx[0];}
}
__global__ void sum_means(const double* sums,int blocks,int count,double* means){
    double y=0.,x=0.;for(int j=threadIdx.x;j<blocks;j+=256){y+=sums[2*j];x+=sums[2*j+1];}
    __shared__ double sy[256],sx[256];sy[threadIdx.x]=y;sx[threadIdx.x]=x;__syncthreads();
    for(int s=128;s;s>>=1){if(threadIdx.x<s){sy[threadIdx.x]+=sy[threadIdx.x+s];sx[threadIdx.x]+=sx[threadIdx.x+s];}__syncthreads();}
    if(threadIdx.x==0){means[0]=sy[0]/count;means[1]=sx[0]/count;}
}
__global__ void crop_dpc(const float* iy,const float* ix,const float* score,int h,int w,int half,
                         const double* mean,double scale,float* oy,float* ox,float* dy,float* dx,float* os){
    int oh=h-2*half,ow=w-2*half,p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=oh*ow)return;
    int q=(p/ow+half)*w+p%ow+half;oy[p]=-iy[q];ox[p]=-ix[q];
    dy[p]=float((double(oy[p])-mean[0])*scale);dx[p]=float((double(ox[p])-mean[1])*scale);os[p]=score[q];
}
}
extern "C" int edge_match(const float* a,const float* b,const float* iy,const float* ix,int h,int w,int d,int r,float* oy,float* ox,float* score,void* stream){
    if(!a||!b||!iy||!ix||!oy||!ox||!score||h<1||w<1||d<1||d>128||r<0||r>5)return cudaErrorInvalidValue;
    edge::match_costs<<<(h*w+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(a,b,iy,ix,h,w,d,r,oy,ox,score);return cudaGetLastError();}
extern "C" int edge_sample_search(const float* image,const float* ref,const float* iy,const float* ix,int h,int w,int r,float* oy,float* ox,float* score,void* stream){
    if(!image||!ref||!iy||!ix||!oy||!ox||!score||h<7||w<7||r<0||r>5)return cudaErrorInvalidValue;
    edge::sample_search<<<(h*w+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(image,ref,iy,ix,h,w,r,oy,ox,score);return cudaGetLastError();}
extern "C" int edge_spline_gpu(const float* iy,const float* ix,int h,int w,int oh,int ow,const double* xd,const double* xf,const int* xl,const double* xt,
    const double* yd,const double* yf,const int* yl,const double* yt,double* xr,double* yr,double* horizontal,float* oy,float* ox,double limit,void* stream){
    if(h<4||w<4||oh<2||ow<2)return cudaErrorInvalidValue;auto s=reinterpret_cast<cudaStream_t>(stream);
    edge::spline_rhs_rows<<<(2*h*(w-2)+255)/256,256,0,s>>>(iy,ix,h,w,xr);
    edge::spline_solve<<<(2*h+127)/128,128,0,s>>>(xr,2*h,w-2,xd,xf);
    edge::spline_rows_eval<<<(2*h*ow+255)/256,256,0,s>>>(iy,ix,h,w,ow,xl,xt,xr,horizontal);
    edge::spline_rhs_cols<<<(2*ow*(h-2)+255)/256,256,0,s>>>(horizontal,h,ow,yr);
    edge::spline_solve<<<(2*ow+127)/128,128,0,s>>>(yr,2*ow,h-2,yd,yf);
    edge::spline_cols_eval<<<(2*oh*ow+255)/256,256,0,s>>>(horizontal,h,oh,ow,yl,yt,yr,oy,ox,limit);return cudaGetLastError();}
extern "C" int edge_stats100(const float* raw,int p,double* mean,double* sd,void* stream){edge::stats100<<<(p+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(raw,p,mean,sd);return cudaGetLastError();}
extern "C" int edge_normalize100(const float* raw,int p,const double* mean,const double* sd,float* out,void* stream){edge::normalize100<<<(100*p+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(raw,p,mean,sd,out);return cudaGetLastError();}
extern "C" int edge_stats100_u16(const unsigned short* raw,int p,double* mean,double* sd,void* stream){edge::stats100<<<(p+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(raw,p,mean,sd);return cudaGetLastError();}
extern "C" int edge_normalize100_u16(const unsigned short* raw,int p,const double* mean,const double* sd,float* out,void* stream){edge::normalize100<<<(100*p+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(raw,p,mean,sd,out);return cudaGetLastError();}
extern "C" int edge_temporal100(const float* raw,int p,const double* mean,const double* sd,int normalize,float* out,void* stream){edge::temporal100<<<(p+31)/32,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(raw,p,mean,sd,normalize,out);return cudaGetLastError();}
extern "C" int edge_temporal100_u16(const unsigned short* raw,int p,const double* mean,const double* sd,int normalize,float* out,void* stream){edge::temporal100<<<(p+31)/32,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(raw,p,mean,sd,normalize,out);return cudaGetLastError();}
extern "C" int edge_spatial(const float* in,int h,int w,int channels,int hwd,float* tmp,float* out,void* stream){int oh=(h+5)/2,ow=(w+5)/2;auto s=reinterpret_cast<cudaStream_t>(stream);
    edge::spatial_x<<<(channels*h*ow+255)/256,256,0,s>>>(in,h,w,ow,channels,hwd,tmp);
    edge::spatial_y<<<(channels*oh*ow+255)/256,256,0,s>>>(tmp,h,oh,ow,channels,hwd,out);return cudaGetLastError();}
extern "C" int edge_spatial_normalized100(const float* raw,int h,int w,const double* mean,const double* sd,int chunk,float* tmp,float* out,void* stream){return edge::spatial_normalized100(raw,h,w,mean,sd,chunk,tmp,out,stream);}
extern "C" int edge_spatial_normalized100_u16(const unsigned short* raw,int h,int w,const double* mean,const double* sd,int chunk,float* tmp,float* out,void* stream){return edge::spatial_normalized100(raw,h,w,mean,sd,chunk,tmp,out,stream);}
extern "C" int edge_transmission(const float* a,const float* b,int n,float* out,void* stream){edge::transmission<<<(n+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(a,b,n,out);return cudaGetLastError();}
extern "C" int edge_transmission_u16(const unsigned short* a,const float* b,int n,float* out,void* stream){edge::transmission<<<(n+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(a,b,n,out);return cudaGetLastError();}
extern "C" int edge_copy_first_u16(const unsigned short* a,int n,float* out,void* stream){edge::copy_first_u16<<<(n+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(a,n,out);return cudaGetLastError();}
extern "C" int edge_post(const float* iy,const float* ix,const float* score,int h,int w,int half,double scale,double* sums,double* means,float* oy,float* ox,float* dy,float* dx,float* os,void* stream){
    int count=(h-2*half)*(w-2*half),blocks=(count+255)/256;auto s=reinterpret_cast<cudaStream_t>(stream);
    edge::crop_sums<<<blocks,256,0,s>>>(iy,ix,h,w,half,sums);edge::sum_means<<<1,256,0,s>>>(sums,blocks,count,means);
    edge::crop_dpc<<<blocks,256,0,s>>>(iy,ix,score,h,w,half,means,scale,oy,ox,dy,dx,os);return cudaGetLastError();}
