"""Isolated experimental WXST CUDA backend. Runs with Jetson system Python.

GPU: db3 pyramid, wrapped 7x7 template/normalization, db2 DWT, exhaustive
SSD/subpixel matching, and Fourier phase integration. CPU: TIFF decoding,
cubic spline propagation and output serialization. Not a published benchmark.
"""
import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline


class NativeWXST:
    def __init__(self):
        self.lib = ctypes.CDLL(str(Path(__file__).resolve().with_name('libwxst_match.so')))
        p, i = ctypes.c_void_p, ctypes.c_int
        self.lib.wxst_match_gpu.argtypes = [p,p,p,p,i,i,i,i,p,p,p,p,p]
        self.lib.wxst_template_gpu.argtypes = [p,i,i,i,p,p]
        self.lib.wxst_downsample_gpu.argtypes = [p,i,i,p,p,p]
        self.lib.wxst_wavelet_gpu.argtypes = [p,i,i,i,i,i,i,p,p]
        for name in ('wxst_match_gpu','wxst_template_gpu','wxst_downsample_gpu','wxst_wavelet_gpu'):
            getattr(self.lib,name).restype=i

    @staticmethod
    def stream():
        return torch.cuda.current_stream().cuda_stream

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError('CUDA returned error code {}'.format(code))

    def downsample(self, image):
        h,w=image.shape;oh,ow=(h+5)//2,(w+5)//2
        tmp=torch.empty((h,ow),device='cuda',dtype=torch.float32)
        out=torch.empty((oh,ow),device='cuda',dtype=torch.float32)
        self.check(self.lib.wxst_downsample_gpu(image.data_ptr(),h,w,tmp.data_ptr(),out.data_ptr(),self.stream()))
        return out

    def features(self,image,n_s=3,cut=1):
        h,w=image.shape;depth=(2*n_s+1)**2
        if depth>121:raise ValueError('Prototype supports templates up to 11x11')
        template=torch.empty((h,w,depth),device='cuda',dtype=torch.float32)
        self.check(self.lib.wxst_template_gpu(image.data_ptr(),h,w,n_s,template.data_ptr(),self.stream()))
        levels=max(0,int(math.floor(math.log2(depth/3))))
        add=2 if depth<=50 else 0
        retained=max(1,min(levels+1,levels+1-cut+add))
        sizes=[];d=depth
        for lv in range(levels):d=(d+3)//2;sizes.append(d)
        out_depth=d+sum(sizes[::-1][:retained-1])
        out=torch.empty((h,w,out_depth),device='cuda',dtype=torch.float32)
        self.check(self.lib.wxst_wavelet_gpu(template.data_ptr(),h,w,depth,levels,retained,out_depth,out.data_ptr(),self.stream()))
        return out,{'input_depth':depth,'wavelet_levels':levels,'retained_blocks':retained,'descriptor_depth':out_depth}

    def match(self,sample,reference,initial_y,initial_x,radius):
        h,w,depth=sample.shape
        out_y=torch.empty((h,w),device='cuda',dtype=torch.float32)
        out_x=torch.empty_like(out_y);score=torch.empty_like(out_y)
        index=torch.empty((h,w),device='cuda',dtype=torch.int32)
        self.check(self.lib.wxst_match_gpu(sample.data_ptr(),reference.data_ptr(),initial_y.data_ptr(),initial_x.data_ptr(),
                                          h,w,depth,radius,out_y.data_ptr(),out_x.data_ptr(),score.data_ptr(),index.data_ptr(),self.stream()))
        return out_y,out_x,score


def spline_guess(field,shape,limit):
    old=field.cpu().numpy()*np.float32(2)
    h,w=old.shape
    fit=RectBivariateSpline(np.arange(h),np.arange(w),old)
    values=fit(np.linspace(0,h-1,shape[0]),np.linspace(0,w-1,shape[1]))
    values=np.clip(np.round(values),-limit,limit).astype(np.float32)
    return torch.from_numpy(values).cuda()


def phase_cpp_convention(dpc_x,dpc_y,pixel,energy):
    h,w=dpc_x.shape
    y=torch.arange(h,device='cuda',dtype=torch.float32)
    x=torch.arange(w,device='cuda',dtype=torch.float32)
    wy=torch.where(y<=h//2,y,y-h)*(2*math.pi/h)
    wx=torch.where(x<=w//2,x,x-w)*(2*math.pi/w)
    denominator=(wy[:,None]**2+wx[None,:]**2).clamp_min(1e-12)
    spectrum=(-1j*wx[None,:]*torch.fft.fft2(dpc_x)-1j*wy[:,None]*torch.fft.fft2(dpc_y))/denominator
    phase=torch.fft.ifft2(spectrum).real
    phase=phase-phase.mean()
    return phase*(-pixel*2*math.pi/(1.2398419843320026e-6/energy))


def solve(sample_np,reference_np,half=10,n_s=3,cut=1,pixel=1.625e-6,distance=.4845,energy=12400):
    if sample_np.shape!=reference_np.shape or sample_np.ndim!=2:raise ValueError('Input shape mismatch')
    if min(sample_np.shape)<=2*half+8:raise ValueError('Input is too small for crop and pyramid')
    backend=NativeWXST()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();stages={}
    sample=torch.from_numpy(np.ascontiguousarray(sample_np,dtype=np.float32)).cuda()
    reference=torch.from_numpy(np.ascontiguousarray(reference_np,dtype=np.float32)).cuda()
    torch.cuda.synchronize();stages['input_to_cuda_s']=time.perf_counter()-started
    t=time.perf_counter()
    coarse_sample=backend.downsample(sample);coarse_reference=backend.downsample(reference)
    coarse_s,meta=backend.features(coarse_sample,n_s,cut)
    coarse_r,_=backend.features(coarse_reference,n_s,cut)
    torch.cuda.synchronize();stages['coarse_features_s']=time.perf_counter()-t
    t=time.perf_counter()
    zero=torch.zeros(coarse_s.shape[:2],device='cuda',dtype=torch.float32)
    dy,dx,_=backend.match(coarse_s,coarse_r,zero,zero,int(math.ceil(half/2)))
    dy=dy.clamp(-half/2,half/2);dx=dx.clamp(-half/2,half/2)
    torch.cuda.synchronize();stages['coarse_match_s']=time.perf_counter()-t
    del coarse_s,coarse_r,coarse_sample,coarse_reference
    t=time.perf_counter()
    guess_y=spline_guess(dy,sample.shape,half);guess_x=spline_guess(dx,sample.shape,half)
    torch.cuda.synchronize();stages['cpu_spline_and_upload_s']=time.perf_counter()-t
    del dy,dx
    t=time.perf_counter()
    fine_s,_=backend.features(sample,n_s,cut);fine_r,_=backend.features(reference,n_s,cut)
    torch.cuda.synchronize();stages['fine_features_s']=time.perf_counter()-t
    t=time.perf_counter()
    dy,dx,score=backend.match(fine_s,fine_r,guess_y,guess_x,4)
    dy=dy.clamp(-half,half);dx=dx.clamp(-half,half)
    torch.cuda.synchronize();stages['fine_match_s']=time.perf_counter()-t
    del fine_s,fine_r,guess_y,guess_x
    t=time.perf_counter()
    dy=-dy[half:-half,half:-half].contiguous();dx=-dx[half:-half,half:-half].contiguous()
    # FP64 only for small reductions/conversion, as in the C++ reference.
    dpc_y=((dy.double()-dy.double().mean())*(pixel/distance)).float()
    dpc_x=((dx.double()-dx.double().mean())*(pixel/distance)).float()
    phase=phase_cpp_convention(dpc_x,dpc_y,pixel,energy)
    torch.cuda.synchronize();stages['gpu_dpc_phase_s']=time.perf_counter()-t
    solver_s=time.perf_counter()-started
    result={name:field.cpu().numpy() for name,field in [('displace_y',dy),('displace_x',dx),('DPC_y',dpc_y),('DPC_x',dpc_x),('phase',phase)]}
    result['match_score_neg_ssd']=score[half:-half,half:-half].cpu().numpy()
    report={'status':'smoke_only','input_shape':list(sample.shape),'output_shape':list(dy.shape),'solver_with_input_upload_s':solver_s,
            'stages':stages,'features':meta,'params':{'n_s':n_s,'search_half':half,'coarse_half':int(math.ceil(half/2)),
            'fine_half':4,'cut':cut,'pyramid_level':1,'pixel_m':pixel,'distance_m':distance,'energy_eV':energy},
            'all_finite':all(bool(np.isfinite(v).all()) for v in result.values()),
            'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'native_cuda_kernel':True,'cpu_stages':['TIFF decoding','cubic spline propagation','output writing'],
            'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'memory_scope':'PyTorch CUDA allocator; excludes CPU arrays, CUDA context and other processes',
            'geometry_status':'historical conversion settings; BB11 optical calibration not newly verified'}
    return result,report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.out.exists():raise FileExistsError('Keep existing result; use a fresh output directory')
    data=np.load(args.input);args.out.mkdir(parents=True)
    result,report=solve(data['sample'],data['reference'])
    np.savez_compressed(args.out/'reconstruction.npz',**result)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['all_finite']:raise RuntimeError('Non-finite output')

if __name__=='__main__':main()
