"""Separate WXST / WSVT edge sessions. CUDA-stream ordering, persistent workspaces.
Only fixed dense/no-DF two-level I=1 configurations are implemented here.
No silent fallback to an algorithm with different normalization or coefficients.
"""
import ctypes as C
import math
import threading
import time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent / '_native'
P,I,D=C.c_void_p,C.c_int,C.c_double

class Native:
    def __init__(self):
        self.lib=C.CDLL(str(HERE/'libedge_kernels.so'))
        signatures={
            'wxst_features_fused_gpu':[P,I,I,I,P,P],
            'edge_match':[P,P,P,P,I,I,I,I,P,P,P,P],
            'edge_sample_search':[P,P,P,P,I,I,I,P,P,P,P],
            'edge_spline_gpu':[P,P,I,I,I,I]+[P]*13+[D,P],
            'edge_stats100':[P,I,P,P,P], 'edge_normalize100':[P,I,P,P,P,P],
            'edge_stats100_u16':[P,I,P,P,P], 'edge_normalize100_u16':[P,I,P,P,P,P],
            'edge_transmission_u16':[P,P,I,P,P], 'edge_copy_first_u16':[P,I,P,P],
            'edge_temporal100':[P,I,P,P,I,P,P],
            'edge_spatial':[P,I,I,I,I,P,P,P],
            'edge_transmission':[P,P,I,P,P],
            'edge_post':[P,P,P,I,I,I,D]+[P]*8,
        }
        for name,args in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=I
    def call(self,name,*args):
        values=[a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args]
        code=getattr(self.lib,name)(*values)
        if code:raise RuntimeError('{} returned CUDA code {}'.format(name,code))

def axis_plan(n,m):
    diagonal=np.full(n-2,4.,np.float64);diagonal[0]=diagonal[-1]=1.
    upper=np.zeros(n-2,np.float64);upper[1:-1]=1.
    factor=np.zeros(n-2,np.float64)
    for j in range(1,n-2):
        factor[j]=(0. if j==n-3 else 1.)/diagonal[j-1]
        diagonal[j]-=factor[j]*upper[j-1]
    coordinates=np.arange(m,dtype=np.float64)*(float(n-1)/float(m-1))
    left=np.minimum(np.floor(coordinates).astype(np.int32),n-2)
    return diagonal,factor,left,coordinates-left

class CPUSpline:
    def __init__(self,shape,target,threads=2):
        self.handle=None
        filename='edge_spline.dll' if __import__('os').name=='nt' else 'libedge_spline.so'
        self.lib=C.CDLL(str(HERE/filename))
        self.lib.edge_spline_create.argtypes=[I]*5;self.lib.edge_spline_create.restype=P
        self.lib.edge_spline_run.argtypes=[P]*5+[D];self.lib.edge_spline_run.restype=I
        self.lib.edge_spline_destroy.argtypes=[P];self.lib.edge_spline_error.restype=C.c_char_p
        self.handle=self.lib.edge_spline_create(*shape,*target,threads)
        if not self.handle:raise RuntimeError(self.lib.edge_spline_error().decode())
    def run_arrays(self,y,x,oy,ox,limit):
        if not self.handle:raise RuntimeError('Closed CPU spline')
        values=[self.handle]+[a.ctypes.data for a in (y,x,oy,ox)]+[limit]
        if self.lib.edge_spline_run(*values):raise RuntimeError(self.lib.edge_spline_error().decode())
    def close(self):
        if getattr(self,'handle',None):self.lib.edge_spline_destroy(self.handle);self.handle=None
    def __del__(self):self.close()

class EdgeSession:
    def __init__(self,shape,kind,front,spline='gpu',device=0,half=10,
                 pixel=1.625e-6,distance=.4845,energy=12400.,cpu_threads=2,input_dtype='float32'):
        if kind not in ('wxst','wsvt'):raise ValueError('Separate supported modes: wxst/wsvt')
        if (kind=='wxst' and front not in ('materialized','fused')) or (kind=='wsvt' and front not in ('baseline','reordered')):raise ValueError('Invalid front end for selected method')
        if spline not in ('cpu','gpu') or half!=10:raise ValueError('This reviewed session supports half10 and cpu/gpu spline only')
        if input_dtype not in ('float32','uint16') or (input_dtype=='uint16' and (kind!='wsvt' or front!='baseline')):
            raise ValueError('uint16 input is supported only by the original-order WSVT backend')
        self.input_dtype=input_dtype
        self.h,self.w=map(int,shape)
        if min(shape)<=2*half+8:raise ValueError('Image too small')
        self.kind,self.front,self.spline=kind,front,spline;self.device=torch.device('cuda',device)
        self.owner_thread=threading.get_ident()
        if not all(math.isfinite(v) and v>0 for v in (pixel,distance,energy)):
            raise ValueError('Pixel size, propagation distance and energy must be positive and finite')
        self.hc,self.wc=(self.h+5)//2,(self.w+5)//2;self.oh,self.ow=self.h-2*half,self.w-2*half
        self.half=half;self.pixel=pixel;self.distance=distance;self.energy=energy
        self.native=Native();self.stream=torch.cuda.current_stream(self.device)
        self.ready=False;self.closed=False;self.in_flight=False;self.b={};self.events={}
        self.frames=100 if kind=='wsvt' else 1;self.depth=63 if kind=='wsvt' else 58
        def alloc(name,shape,dtype=torch.float32):
            self.b[name]=torch.empty(shape,device=self.device,dtype=dtype);return self.b[name]
        if input_dtype=='uint16':alloc('raw',(self.frames,self.h,self.w*2),torch.uint8)
        else:alloc('raw',(self.frames,self.h,self.w) if kind=='wsvt' else (self.h,self.w))
        alloc('ref_first',(self.h,self.w));alloc('transmission',(self.h,self.w))
        alloc('ref_fine',(self.h,self.w,self.depth));alloc('ref_coarse',(self.hc,self.wc,self.depth))
        if kind=='wsvt' or front=='materialized':
            alloc('sample_fine',(self.h,self.w,self.depth));alloc('sample_coarse',(self.hc,self.wc,self.depth))
        if kind=='wxst':
            alloc('coarse_image',(self.hc,self.wc));alloc('spatial_tmp',(self.h,self.wc))
        else:
            alloc('mean',(self.h,self.w),torch.float64);alloc('std',(self.h,self.w),torch.float64)
            channels=100 if front=='baseline' else 63
            alloc('spatial_tmp',(channels,self.h,self.wc))
            if front=='baseline':alloc('normalized',(100,self.h,self.w));alloc('coarse_stack',(100,self.hc,self.wc))
        for level,shape_l in [('coarse',(self.hc,self.wc)),('fine',(self.h,self.w))]:
            for field in ('y','x','score'):alloc(level+'_'+field,shape_l)
        self.b['zero']=torch.zeros((self.hc,self.wc),device=self.device)
        for name in ('guess_y','guess_x'):alloc(name,(self.h,self.w))
        for name in ('displace_y','displace_x','DPC_y','DPC_x','phase','match_score_neg_ssd'):alloc(name,(self.oh,self.ow))
        alloc('sums',(2*((self.oh*self.ow+255)//256),),torch.float64);alloc('means',(2,),torch.float64)
        if spline=='gpu':
            for prefix,n,m in [('sx',self.wc,self.w),('sy',self.hc,self.h)]:
                for suffix,array in zip(('diag','factor','left','t'),axis_plan(n,m)):
                    self.b[prefix+'_'+suffix]=torch.from_numpy(array).to(self.device)
            alloc('spline_rhs_x',(self.wc-2,2*self.hc),torch.float64)
            alloc('spline_rhs_y',(self.hc-2,2*self.w),torch.float64)
            alloc('spline_horizontal',(2,self.hc,self.w),torch.float64)
        else:
            self.cpu_plan=CPUSpline((self.hc,self.wc),(self.h,self.w),cpu_threads)
            self.host={name:torch.empty(shape,device='cpu',pin_memory=True) for name,shape in
                       [('y',(self.hc,self.wc)),('x',(self.hc,self.wc)),('oy',(self.h,self.w)),('ox',(self.h,self.w))]}
        # Fixed phase frequencies and output buffers. PyTorch/cuFFT caches FFT plans;
        # first invocation is explicitly warm-up, not a cold-start performance claim.
        yy=torch.arange(self.oh,device=self.device,dtype=torch.float32)
        xx=torch.arange(self.ow,device=self.device,dtype=torch.float32)
        self.wy=torch.where(yy<=self.oh//2,yy,yy-self.oh)*(2*math.pi/self.oh)
        self.wx=torch.where(xx<=self.ow//2,xx,xx-self.ow)*(2*math.pi/self.ow)
        self.den=(self.wy[:,None]**2+self.wx[None,:]**2).clamp_min(1e-12)
        for name in ('fft_x','fft_y','spectrum','ifft'):alloc(name,(self.oh,self.ow),torch.complex64)
        self.phase_scale=-pixel*2*math.pi/(1.2398419843320026e-6/energy)

    def _call(self,name,*args):self.native.call(name,*args,self.stream.cuda_stream)
    def _available(self):
        if threading.get_ident()!=self.owner_thread:
            raise RuntimeError('A session belongs to its creating thread; use separate sessions for independent workers')
        if self.closed or self.in_flight:raise RuntimeError('Session unavailable')
    def _validate(self,array):
        expected=(100,self.h,self.w) if self.kind=='wsvt' else (self.h,self.w)
        if array.shape!=expected or array.dtype!=np.dtype(self.input_dtype) or not array.flags.c_contiguous:
            raise ValueError('Input must be contiguous {} with shape {}'.format(self.input_dtype,expected))
        # Every uint16 value is finite by construction; keep float validation intact.
        if self.input_dtype=='float32' and not np.isfinite(array).all():raise ValueError('Non-finite input')
    def _upload(self,array):
        view=array.view(np.uint8) if self.input_dtype=='uint16' else array
        self.b['raw'].copy_(torch.from_numpy(view))
    def _stage(self,name,fn):
        if name not in self.events:self.events[name]=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
        start,end=self.events[name];start.record(self.stream);fn();end.record(self.stream)
    def _spatial(self,input,channels,hwd,output):
        self._call('edge_spatial',input,self.h,self.w,channels,int(hwd),self.b['spatial_tmp'],output)
    def _features(self,fine,coarse):
        b=self.b
        if self.kind=='wxst':
            self._spatial(b['raw'],1,False,b['coarse_image'])
            self._call('wxst_features_fused_gpu',b['raw'],self.h,self.w,2,fine)
            self._call('wxst_features_fused_gpu',b['coarse_image'],self.hc,self.wc,2,coarse)
        else:
            suffix='_u16' if self.input_dtype=='uint16' else ''
            self._call('edge_stats100'+suffix,b['raw'],self.h*self.w,b['mean'],b['std'])
            if self.front=='reordered':
                self._call('edge_temporal100',b['raw'],self.h*self.w,b['mean'],b['std'],1,fine)
                self._spatial(fine,63,True,coarse)
            else:
                self._call('edge_normalize100'+suffix,b['raw'],self.h*self.w,b['mean'],b['std'],b['normalized'])
                self._spatial(b['normalized'],100,False,b['coarse_stack'])
                self._call('edge_temporal100',b['normalized'],self.h*self.w,b['mean'],b['std'],0,fine)
                self._call('edge_temporal100',b['coarse_stack'],self.hc*self.wc,b['mean'],b['std'],0,coarse)
    def prepare_reference(self,array):
        self._available()
        self._validate(array);self.ready=False;t=time.perf_counter()
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream):
            self._upload(array)
            if self.input_dtype=='uint16':self._call('edge_copy_first_u16',self.b['raw'],self.h*self.w,self.b['ref_first'])
            else:self.b['ref_first'].copy_(self.b['raw'][0] if self.kind=='wsvt' else self.b['raw'])
            self._features(self.b['ref_fine'],self.b['ref_coarse']);self.stream.synchronize()
        self.reference_prepare_s=time.perf_counter()-t;self.ready=True
        return self.reference_prepare_s
    def _spline(self):
        b=self.b
        if self.spline=='gpu':
            self._call('edge_spline_gpu',b['coarse_y'],b['coarse_x'],self.hc,self.wc,self.h,self.w,
                       *[b[p+'_'+s] for p in ('sx','sy') for s in ('diag','factor','left','t')],
                       b['spline_rhs_x'],b['spline_rhs_y'],b['spline_horizontal'],b['guess_y'],b['guess_x'],float(self.half))
        else:
            self.host['y'].copy_(b['coarse_y']);self.host['x'].copy_(b['coarse_x'])
            self.cpu_plan.run_arrays(*[self.host[n].numpy() for n in ('y','x','oy','ox')],float(self.half))
            b['guess_y'].copy_(self.host['oy'],non_blocking=True);b['guess_x'].copy_(self.host['ox'],non_blocking=True)
    def _phase(self):
        b=self.b
        torch.fft.fft2(b['DPC_x'],out=b['fft_x']);torch.fft.fft2(b['DPC_y'],out=b['fft_y'])
        # Reuse persistent spectrum; this expression still has temporary pointwise
        # tensors, reported honestly rather than claiming zero allocations everywhere.
        b['spectrum'].copy_((-1j*self.wx[None,:]*b['fft_x']-1j*self.wy[:,None]*b['fft_y'])/self.den)
        torch.fft.ifft2(b['spectrum'],out=b['ifft'])
        real=b['ifft'].real;torch.sub(real,real.mean(),out=b['phase']);b['phase'].mul_(self.phase_scale)
    def solve(self,array,download=True):
        self._available()
        if not self.ready:raise RuntimeError('Prepare reference before solving')
        self._validate(array);self.in_flight=True;b=self.b;started=time.perf_counter();spline_wall=0.
        try:
            with torch.cuda.device(self.device),torch.cuda.stream(self.stream):
                self._stage('input',lambda:self._upload(array))
                self._stage('transmission',lambda:self._call('edge_transmission'+('_u16' if self.input_dtype=='uint16' else ''),b['raw'],b['ref_first'],self.h*self.w,b['transmission']))
                if self.kind=='wxst' and self.front=='fused':
                    self._stage('sample_front',lambda:self._spatial(b['raw'],1,False,b['coarse_image']))
                else:self._stage('sample_front',lambda:self._features(b['sample_fine'],b['sample_coarse']))
                def matching(level,h,w,r,iy,ix):
                    if self.kind=='wxst' and self.front=='fused':
                        image=b['coarse_image'] if level=='coarse' else b['raw']
                        self._call('edge_sample_search',image,b['ref_'+level],iy,ix,h,w,r,b[level+'_y'],b[level+'_x'],b[level+'_score'])
                    else:self._call('edge_match',b['sample_'+level],b['ref_'+level],iy,ix,h,w,self.depth,r,b[level+'_y'],b[level+'_x'],b[level+'_score'])
                    b[level+'_y'].clamp_(-self.half/(2 if level=='coarse' else 1),self.half/(2 if level=='coarse' else 1))
                    b[level+'_x'].clamp_(-self.half/(2 if level=='coarse' else 1),self.half/(2 if level=='coarse' else 1))
                self._stage('coarse_match',lambda:matching('coarse',self.hc,self.wc,5,b['zero'],b['zero']))
                spline_start=time.perf_counter();self._stage('spline',self._spline);spline_wall=time.perf_counter()-spline_start
                self._stage('fine_match',lambda:matching('fine',self.h,self.w,4,b['guess_y'],b['guess_x']))
                self._stage('DPC',lambda:self._call('edge_post',b['fine_y'],b['fine_x'],b['fine_score'],self.h,self.w,self.half,
                            self.pixel/self.distance,b['sums'],b['means'],b['displace_y'],b['displace_x'],b['DPC_y'],b['DPC_x'],b['match_score_neg_ssd']))
                self._stage('phase',self._phase);self.stream.synchronize()
                solver=time.perf_counter()-started
                outputs={n:b[n].cpu().numpy().copy() for n in ('displace_y','displace_x','DPC_y','DPC_x','phase','transmission','match_score_neg_ssd')} if download else None
            info={'kind':self.kind,'front':self.front,'spline':self.spline,'input_dtype':self.input_dtype,'solver_s':solver,'call_with_download_s':time.perf_counter()-started,
                  'reference_prepare_s':self.reference_prepare_s,'gpu_event_s':{n:a.elapsed_time(z)/1000 for n,(a,z) in self.events.items()},
                  'spline_enqueue_or_cpu_handoff_wall_s':spline_wall,'persistent_device_bytes':sum(v.numel()*v.element_size() for v in b.values())+self.wx.numel()*4+self.wy.numel()*4+self.den.numel()*4,
                  'scope':'sample solver with reference precomputed; pair latency must add reference preparation; no disk I/O',
                  'outputs_owned_copies':download,'geometry_status':'explicit session settings; calibration is not inferred by runtime'}
            return outputs,info
        finally:self.in_flight=False
    def close(self):
        if threading.get_ident()!=self.owner_thread:raise RuntimeError('Close the session on its creating thread')
        if self.in_flight:raise RuntimeError('Cannot close during solve')
        if not self.closed:
            self.stream.synchronize()
            if hasattr(self,'cpu_plan'):self.cpu_plan.close()
            self.b.clear();self.events.clear()
            self.wx=self.wy=self.den=None
            if hasattr(self,'host'):self.host.clear()
            self.closed=True;self.ready=False
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class WXSTEdgeSession(EdgeSession):
    def __init__(self,shape,front='fused',**kwargs):super().__init__(shape,'wxst',front,**kwargs)
class WSVTEdgeSession(EdgeSession):
    def __init__(self,shape,front='baseline',**kwargs):super().__init__(shape,'wsvt',front,**kwargs)
