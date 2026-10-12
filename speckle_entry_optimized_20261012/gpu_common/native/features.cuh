// Independent feature-fusion experiment. Baseline translation unit stays intact.
#include "primitives.cuh"

// One thread per pixel: preserve the original ordered float32 normalization.
__global__ void fused_features_thread(const float* image, int h, int w, float* out) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=h*w)return;
    float a[49],b[26];float sum=0,square=0;int k=0;
    for(int dy=-3;dy<=3;++dy)for(int dx=-3;dx<=3;++dx){
        const float v=image[((p/w-dy+h)%h)*w+(p%w-dx+w)%w];
        a[k++]=v;sum+=v;square+=v*v;
    }
    const float mean=sum*(1.0f/49.0f);
    const float inv=1.0f/(sqrtf(fmaxf(0.0f,square*(1.0f/49.0f)-mean*mean))+1e-6f);
    for(k=0;k<49;++k)a[k]=(a[k]-mean)*inv;
    const int sizes[4]={26,14,8,5},offsets[4]={32,18,10,5};
    float* cur=a;float* next=b;int d=49;
    for(int lv=0;lv<4;++lv){
        for(int oc=0;oc<sizes[lv];++oc){
            float ca=0,cd=0;
            for(int j=0;j<4;++j){const int ic=2*oc+j-2;
                if(ic>=0&&ic<d){ca+=db2_lo[j]*cur[ic];cd+=db2_hi[j]*cur[ic];}}
            next[oc]=ca;out[static_cast<size_t>(p)*58+offsets[lv]+oc]=cd;
        }
        float* tmp=cur;cur=next;next=tmp;d=sizes[lv];
    }
    for(k=0;k<5;++k)out[static_cast<size_t>(p)*58+k]=cur[k];
}

// One warp per pixel: coefficients are adjacent across lanes. Ordered mean/sum
// in lane 0 keeps the baseline's cancellation-sensitive normalization unchanged.
__global__ void fused_features_warp(const float* image,int h,int w,float* out){
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const int p=blockIdx.x*(blockDim.x/32)+warp;
    if(p>=h*w)return;
    __shared__ float a[4][64],b[4][64];
    for(int k=lane;k<49;k+=32){const int dy=k/7-3,dx=k%7-3;
        a[warp][k]=image[((p/w-dy+h)%h)*w+(p%w-dx+w)%w];}
    __syncwarp(MASK);
    float mean=0,inv=0;
    if(lane==0){float sum=0,square=0;
        for(int k=0;k<49;++k){const float v=a[warp][k];sum+=v;square+=v*v;}
        mean=sum*(1.0f/49.0f);
        inv=1.0f/(sqrtf(fmaxf(0.0f,square*(1.0f/49.0f)-mean*mean))+1e-6f);}
    mean=__shfl_sync(MASK,mean,0);inv=__shfl_sync(MASK,inv,0);
    for(int k=lane;k<49;k+=32)a[warp][k]=(a[warp][k]-mean)*inv;
    __syncwarp(MASK);
    const int sizes[4]={26,14,8,5},offsets[4]={32,18,10,5};
    float* cur=a[warp];float* next=b[warp];int d=49;
    for(int lv=0;lv<4;++lv){
        if(lane<sizes[lv]){float ca=0,cd=0;
            for(int j=0;j<4;++j){const int ic=2*lane+j-2;
                if(ic>=0&&ic<d){ca+=db2_lo[j]*cur[ic];cd+=db2_hi[j]*cur[ic];}}
            next[lane]=ca;out[static_cast<size_t>(p)*58+offsets[lv]+lane]=cd;}
        __syncwarp(MASK);
        float* tmp=cur;cur=next;next=tmp;d=sizes[lv];
    }
    if(lane<5)out[static_cast<size_t>(p)*58+lane]=cur[lane];
}

extern "C" int wxst_features_fused_gpu(const float* image,int h,int w,int mode,float* out,void* stream){
    if(!image||!out||h<=6||w<=6||(mode!=1&&mode!=2))return cudaErrorInvalidValue;
    const auto s=reinterpret_cast<cudaStream_t>(stream);
    if(mode==1)fused_features_thread<<<(h*w+127)/128,128,0,s>>>(image,h,w,out);
    else fused_features_warp<<<(h*w+3)/4,128,0,s>>>(image,h,w,out);
    return cudaGetLastError();
}
