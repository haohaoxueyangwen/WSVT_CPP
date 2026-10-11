// Experimental Orin backend: exhaustive HWD SSD and WXST subpixel refinement.
// Includes wrapped spatial templates, db3 pyramid downsampling and db2 DWT.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

static constexpr unsigned MASK = 0xffffffffu;

__device__ float candidate_ssd(const float* sample, const float* reference,
                             int pixel, int ry, int rx, int h, int w, int depth) {
    const int lane = threadIdx.x & 31;
    const bool valid = ry >= 0 && ry < h && rx >= 0 && rx < w;
    float cost = 0.0f;
    for (int k = lane; k < depth; k += 32) {
        const float a = sample[static_cast<size_t>(pixel) * depth + k];
        const float b = valid ? reference[(static_cast<size_t>(ry) * w + rx) * depth + k] : 0.0f;
        const float diff = a - b;
        cost += diff * diff;
    }
    for (int offset = 16; offset; offset >>= 1)
        cost += __shfl_down_sync(MASK, cost, offset);
    return cost;
}

__global__ void find_best(const float* sample, const float* reference,
                          const float* init_y, const float* init_x,
                          int h, int w, int depth, int radius,
                          int* best_index, float* best_score) {
    const int lane = threadIdx.x & 31;
    const int pixel = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (pixel >= h * w) return;
    const int y0 = pixel / w + static_cast<int>(init_y[pixel]);
    const int x0 = pixel % w + static_cast<int>(init_x[pixel]);
    const int side = 2 * radius + 1;
    float best = CUDART_INF_F;
    int winner = 0;
    for (int cy = 0; cy < side; ++cy) {
        for (int cx = 0; cx < side; ++cx) {
            const float cost = candidate_ssd(sample, reference, pixel,
                                            y0 + cy - radius, x0 + cx - radius, h, w, depth);
            if (lane == 0 && cost < best) {
                best = cost;
                winner = cy * side + cx;
            }
        }
    }
    if (lane == 0) { best_index[pixel] = winner; best_score[pixel] = -best; }
}

__global__ void refine(const float* sample, const float* reference,
                       const float* init_y, const float* init_x,
                       int h, int w, int depth, int radius,
                       const int* best_index, float* out_y, float* out_x) {
    const int lane = threadIdx.x & 31;
    const int pixel = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (pixel >= h * w) return;
    const int side = 2 * radius + 1;
    const int by = best_index[pixel] / side, bx = best_index[pixel] % side;
    const int y0 = pixel / w + static_cast<int>(init_y[pixel]);
    const int x0 = pixel % w + static_cast<int>(init_x[pixel]);
    float patch[9];
    for (int py = -1; py <= 1; ++py) {
        for (int px = -1; px <= 1; ++px) {
            const int cy = max(0, min(side - 1, by + py));
            const int cx = max(0, min(side - 1, bx + px));
            const float cost = candidate_ssd(sample, reference, pixel,
                                            y0 + cy - radius, x0 + cx - radius, h, w, depth);
            if (lane == 0) patch[(py + 1) * 3 + px + 1] = -cost;
        }
    }
    if (lane == 0) {
        // Same finite-difference Hessian, zero determinant and clipping rules
        // as WXST::displace_wavelet; no positivity guard is introduced here.
        const float gy = (patch[7] - patch[1]) * 0.5f;
        const float gx = (patch[5] - patch[3]) * 0.5f;
        const float dyy = patch[7] + patch[1] - 2.0f * patch[4];
        const float dxx = patch[5] + patch[3] - 2.0f * patch[4];
        const float dxy = (patch[8] - patch[6] - patch[2] + patch[0]) * 0.25f;
        const float denom = dxx * dyy - dxy * dxy;
        const float inv = denom != 0.0f ? 1.0f / denom : 0.0f;
        const float dx = -(dyy * gx - dxy * gy) * inv;
        const float dy = -(dxx * gy - dxy * gx) * inv;
        out_x[pixel] = fminf(radius, fmaxf(-radius, bx - radius + dx)) + init_x[pixel];
        out_y[pixel] = fminf(radius, fmaxf(-radius, by - radius + dy)) + init_y[pixel];
    }
}

extern "C" int wxst_match_gpu(const float* sample, const float* reference,
                              const float* init_y, const float* init_x,
                              int h, int w, int depth, int radius,
                              float* out_y, float* out_x, float* best_score,
                              int* best_index, void* stream_ptr) {
    if (!sample || !reference || !init_y || !init_x || !out_y || !out_x ||
        !best_score || !best_index || h <= 0 || w <= 0 || depth <= 0 || radius < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    constexpr int threads = 128;
    const int blocks = (h * w + threads / 32 - 1) / (threads / 32);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    find_best<<<blocks, threads, 0, stream>>>(sample, reference, init_y, init_x,
                                            h, w, depth, radius, best_index, best_score);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    refine<<<blocks, threads, 0, stream>>>(sample, reference, init_y, init_x,
                                         h, w, depth, radius, best_index, out_y, out_x);
    return static_cast<int>(cudaGetLastError());
}

__global__ void template_normalize(const float* image, int h, int w, int radius, float* out) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= h*w) return;
    const int side = 2*radius+1, depth=side*side;
    float sum=0.0f, square=0.0f;
    float* dst=out+static_cast<size_t>(pixel)*depth;
    int k=0;
    for(int dy=-radius;dy<=radius;++dy) for(int dx=-radius;dx<=radius;++dx) {
        const int y=(pixel/w-dy+h)%h, x=(pixel%w-dx+w)%w;
        const float v=image[y*w+x]; dst[k++]=v; sum+=v; square+=v*v;
    }
    const float inv_depth=1.0f/float(depth);
    const float mean=sum*inv_depth, var=square*inv_depth-mean*mean;
    const float inv=1.0f/(sqrtf(fmaxf(0.0f,var))+1e-6f);
    for(k=0;k<depth;++k) dst[k]=(dst[k]-mean)*inv;
}

__constant__ float db3_reversed[6]={0.3326705530f,0.8068915093f,0.4598775021f,
                                  -0.1350110200f,-0.0854412739f,0.0352262919f};
__global__ void downsample_row(const float* in,int h,int w,int ow,float* out) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=h*ow)return;
    float sum=0;
    for(int j=0;j<6;++j){const int x=2*(p%ow)+j-4;if(x>=0&&x<w)sum+=db3_reversed[j]*in[(p/ow)*w+x];}
    out[p]=sum;
}
__global__ void downsample_col(const float* in,int h,int ow,int oh,float* out) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=oh*ow)return;
    float sum=0;
    for(int j=0;j<6;++j){const int y=2*(p/ow)+j-4;if(y>=0&&y<h)sum+=db3_reversed[j]*in[y*ow+p%ow];}
    out[p]=sum;
}

__constant__ float db2_lo[4]={0.4829629131f,0.8365163037f,0.2241438680f,-0.1294095226f};
__constant__ float db2_hi[4]={-0.1294095226f,-0.2241438680f,0.8365163037f,-0.4829629131f};
__global__ void wavelet_db2(const float* in,int pixels,int depth,int levels,int retained,
                            int out_depth,float* out) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=pixels)return;
    float a[121],b[121]; int sizes[8],offsets[8];
    int d=depth;
    for(int lv=0;lv<levels;++lv){d=(d+3)/2;sizes[lv]=d;offsets[lv]=-1;}
    int out_offset=d;
    const int start=levels+1-retained;
    for(int lv=levels-1;lv>=start;--lv){offsets[lv]=out_offset;out_offset+=sizes[lv];}
    float* cur=a;float* next=b;
    for(int k=0;k<depth;++k)cur[k]=in[static_cast<size_t>(p)*depth+k];
    d=depth;
    for(int lv=0;lv<levels;++lv){
        for(int oc=0;oc<sizes[lv];++oc){
            float ca=0,cd=0;
            for(int j=0;j<4;++j){const int ic=oc*2+j-2;if(ic>=0&&ic<d){ca+=db2_lo[j]*cur[ic];cd+=db2_hi[j]*cur[ic];}}
            next[oc]=ca;
            if(offsets[lv]>=0)out[static_cast<size_t>(p)*out_depth+offsets[lv]+oc]=cd;
        }
        float* tmp=cur;cur=next;next=tmp;d=sizes[lv];
    }
    for(int k=0;k<d;++k)out[static_cast<size_t>(p)*out_depth+k]=cur[k];
}

extern "C" int wxst_template_gpu(const float* image,int h,int w,int radius,float* out,void* stream) {
    if(h<=2*radius||w<=2*radius||radius<0||radius>5)return cudaErrorInvalidValue;
    template_normalize<<<(h*w+127)/128,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(image,h,w,radius,out);
    return cudaGetLastError();
}
extern "C" int wxst_downsample_gpu(const float* image,int h,int w,float* tmp,float* out,void* stream) {
    const int oh=(h+5)/2,ow=(w+5)/2;const auto s=reinterpret_cast<cudaStream_t>(stream);
    downsample_row<<<(h*ow+127)/128,128,0,s>>>(image,h,w,ow,tmp);
    auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
    downsample_col<<<(oh*ow+127)/128,128,0,s>>>(tmp,h,ow,oh,out);
    return cudaGetLastError();
}
extern "C" int wxst_wavelet_gpu(const float* in,int h,int w,int depth,int levels,int retained,
                                int out_depth,float* out,void* stream) {
    if(depth<1||depth>121||levels<0||levels>7||retained<1||retained>levels+1)return cudaErrorInvalidValue;
    wavelet_db2<<<(h*w+63)/64,64,0,reinterpret_cast<cudaStream_t>(stream)>>>(in,h*w,depth,levels,retained,out_depth,out);
    return cudaGetLastError();
}

#ifdef WXST_MATCH_SMOKE_MAIN
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
static float cpu_cost(const std::vector<float>& a, const std::vector<float>& b,
                      int pixel, int ry, int rx, int h, int w, int depth) {
    float s = 0.0f;
    const bool valid = ry >= 0 && ry < h && rx >= 0 && rx < w;
    for (int k = 0; k < depth; ++k) {
        const float diff = a[static_cast<size_t>(pixel) * depth + k] -
            (valid ? b[(static_cast<size_t>(ry) * w + rx) * depth + k] : 0.0f);
        s += diff * diff;
    }
    return s;
}
int main() {
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
    std::printf("{\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"tests\":[",
                prop.name, prop.major, prop.minor);
    bool first = true, all_pass = true;
    for (int test = 0; test < 5; ++test) {
        const int h = 17, w = 19, n = h * w;
        const int depths[] = {7, 49, 84, 100, 49};
        const int depth = depths[test], radius = test == 3 ? 10 : 3;
        std::mt19937 rng(17 + test); std::uniform_real_distribution<float> random(-1, 1);
        std::vector<float> a(n * depth), b(n * depth), iy(n), ix(n), oy(n), ox(n), scores(n);
        std::vector<int> indices(n);
        for (auto& x : a) x = test == 4 ? 0.0f : random(rng);
        for (auto& x : b) x = test == 4 ? 0.0f : random(rng);
        for (int p = 0; p < n; ++p) { iy[p] = (p % 3) - 1 + 0.25f; ix[p] = (p % 5) - 2 - 0.25f; }
        float *da, *db, *dy, *dx, *doy, *dox, *ds; int* di;
        check(cudaMalloc(&da, a.size() * 4)); check(cudaMalloc(&db, b.size() * 4));
        check(cudaMalloc(&dy, n * 4)); check(cudaMalloc(&dx, n * 4));
        check(cudaMalloc(&doy, n * 4)); check(cudaMalloc(&dox, n * 4));
        check(cudaMalloc(&ds, n * 4)); check(cudaMalloc(&di, n * 4));
        check(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dy, iy.data(), n * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx, ix.data(), n * 4, cudaMemcpyHostToDevice));
        check(static_cast<cudaError_t>(wxst_match_gpu(da, db, dy, dx, h, w, depth, radius, doy, dox, ds, di, nullptr)));
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(oy.data(), doy, n * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(ox.data(), dox, n * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(scores.data(), ds, n * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(indices.data(), di, n * 4, cudaMemcpyDeviceToHost));
        int mismatches = 0; float max_cost_error = 0, max_disp_error = 0;
        const int side = 2 * radius + 1;
        for (int p = 0; p < n; ++p) {
            const int y0 = p / w + static_cast<int>(iy[p]), x0 = p % w + static_cast<int>(ix[p]);
            float best = std::numeric_limits<float>::infinity(); int bi = 0;
            for (int y = 0; y < side; ++y) for (int x = 0; x < side; ++x) {
                const float cost = cpu_cost(a, b, p, y0 + y - radius, x0 + x - radius, h, w, depth);
                if (cost < best) { best = cost; bi = y * side + x; }
            }
            mismatches += bi != indices[p];
            max_cost_error = std::max(max_cost_error, std::fabs(best + scores[p]));
            float patch[9];
            for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
                const int cy = std::max(0, std::min(side - 1, bi / side + y));
                const int cx = std::max(0, std::min(side - 1, bi % side + x));
                patch[(y+1)*3+x+1] = -cpu_cost(a,b,p,y0+cy-radius,x0+cx-radius,h,w,depth);
            }
            const float gy=(patch[7]-patch[1])*0.5f, gx=(patch[5]-patch[3])*0.5f;
            const float dyy=patch[7]+patch[1]-2*patch[4], dxx=patch[5]+patch[3]-2*patch[4];
            const float dxy=(patch[8]-patch[6]-patch[2]+patch[0])*0.25f;
            const float denom=dxx*dyy-dxy*dxy, inv=denom!=0 ? 1/denom : 0;
            const float cx=std::max(float(-radius),std::min(float(radius),float(bi%side-radius)-(dyy*gx-dxy*gy)*inv))+ix[p];
            const float cy=std::max(float(-radius),std::min(float(radius),float(bi/side-radius)-(dxx*gy-dxy*gx)*inv))+iy[p];
            max_disp_error=std::max(max_disp_error,std::max(std::fabs(cx-ox[p]),std::fabs(cy-oy[p])));
            if (!std::isfinite(ox[p]) || !std::isfinite(oy[p])) all_pass=false;
        }
        const bool pass=mismatches==0 && max_cost_error<1e-3f && max_disp_error<1e-3f;
        all_pass &= pass;
        std::printf("%s{\"depth\":%d,\"radius\":%d,\"all_zero\":%s,\"integer_mismatches\":%d,\"max_cost_abs_error\":%.9g,\"max_displacement_abs_error_px\":%.9g,\"pass\":%s}",
                    first?"":",",depth,radius,test==4?"true":"false",mismatches,max_cost_error,max_disp_error,pass?"true":"false");
        first=false;
        for (void* p : {static_cast<void*>(da),static_cast<void*>(db),static_cast<void*>(dy),static_cast<void*>(dx),static_cast<void*>(doy),static_cast<void*>(dox),static_cast<void*>(ds),static_cast<void*>(di)}) check(cudaFree(p));
    }
    std::printf("],\"pass\":%s}\n",all_pass?"true":"false");
    return all_pass?0:1;
}
#endif
