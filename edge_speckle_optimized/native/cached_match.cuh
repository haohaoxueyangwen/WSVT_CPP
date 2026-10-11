// Second isolated candidate: retain sample features across all candidates and
// perform best-candidate search plus 3x3 refinement in one kernel.
#include "features.cuh"

__device__ __forceinline__ float cached_cost(float a0,float a1,float a2,float a3,
    const float* reference,int ry,int rx,int h,int w,int depth){
    const int lane=threadIdx.x&31;
    const bool valid=ry>=0&&ry<h&&rx>=0&&rx<w;
    const size_t offset=valid?(static_cast<size_t>(ry)*w+rx)*depth:0;
    float cost=0;
    if(lane<depth){const float diff=a0-(valid?reference[offset+lane]:0.0f);cost+=diff*diff;}
    if(lane+32<depth){const float diff=a1-(valid?reference[offset+lane+32]:0.0f);cost+=diff*diff;}
    if(lane+64<depth){const float diff=a2-(valid?reference[offset+lane+64]:0.0f);cost+=diff*diff;}
    if(lane+96<depth){const float diff=a3-(valid?reference[offset+lane+96]:0.0f);cost+=diff*diff;}
    for(int shift=16;shift;shift>>=1)cost+=__shfl_down_sync(MASK,cost,shift);
    return cost;
}

__global__ void match_cached(const float* sample,const float* reference,const float* iy,const float* ix,
    int h,int w,int depth,int radius,float* oy,float* ox,float* score,int* index){
    const int lane=threadIdx.x&31,p=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(p>=h*w)return;
    const size_t offset=static_cast<size_t>(p)*depth;
    const float a0=lane<depth?sample[offset+lane]:0;
    const float a1=lane+32<depth?sample[offset+lane+32]:0;
    const float a2=lane+64<depth?sample[offset+lane+64]:0;
    const float a3=lane+96<depth?sample[offset+lane+96]:0;
    const int y0=p/w+static_cast<int>(iy[p]),x0=p%w+static_cast<int>(ix[p]),side=2*radius+1;
    float best=CUDART_INF_F;int winner=0;
    for(int cy=0;cy<side;++cy)for(int cx=0;cx<side;++cx){
        const float cost=cached_cost(a0,a1,a2,a3,reference,y0+cy-radius,x0+cx-radius,h,w,depth);
        if(lane==0&&cost<best){best=cost;winner=cy*side+cx;}}
    winner=__shfl_sync(MASK,winner,0);
    const int by=winner/side,bx=winner%side;float patch[9];
    for(int py=-1;py<=1;++py)for(int px=-1;px<=1;++px){
        const int cy=max(0,min(side-1,by+py)),cx=max(0,min(side-1,bx+px));
        const float cost=cached_cost(a0,a1,a2,a3,reference,y0+cy-radius,x0+cx-radius,h,w,depth);
        if(lane==0)patch[(py+1)*3+px+1]=-cost;}
    if(lane==0){
        const float gy=(patch[7]-patch[1])*0.5f,gx=(patch[5]-patch[3])*0.5f;
        const float dyy=patch[7]+patch[1]-2.0f*patch[4],dxx=patch[5]+patch[3]-2.0f*patch[4];
        const float dxy=(patch[8]-patch[6]-patch[2]+patch[0])*0.25f;
        const float denom=dxx*dyy-dxy*dxy,inv=denom!=0.0f?1.0f/denom:0.0f;
        const float dx=-(dyy*gx-dxy*gy)*inv,dy=-(dxx*gy-dxy*gx)*inv;
        ox[p]=fminf(radius,fmaxf(-radius,bx-radius+dx))+ix[p];
        oy[p]=fminf(radius,fmaxf(-radius,by-radius+dy))+iy[p];
        score[p]=-best;index[p]=winner;
    }
}
extern "C" int wxst_match_cached_gpu(const float* sample,const float* reference,const float* iy,const float* ix,
    int h,int w,int depth,int radius,float* oy,float* ox,float* score,int* index,void* stream){
    if(depth>128)return wxst_match_gpu(sample,reference,iy,ix,h,w,depth,radius,oy,ox,score,index,stream);
    if(!sample||!reference||!iy||!ix||!oy||!ox||!score||!index||h<=0||w<=0||depth<=0||radius<0)return cudaErrorInvalidValue;
    match_cached<<<(h*w+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(sample,reference,iy,ix,h,w,depth,radius,oy,ox,score,index);
    return cudaGetLastError();
}
