"""Optional NumPy/ctypes binding. All reconstruction and FFT execute in C++/CUDA."""
import ctypes as C
import numpy as np
class Config(C.Structure):
    _fields_=[(n,C.c_int) for n in ('height','width','mode','input_u16','low_memory','spatial_chunk','device','mapped_input')]+[(n,C.c_double) for n in ('pixel_m','distance_m','energy_ev')]
class Timing(C.Structure):
    _fields_=[(n,C.c_double) for n in ('reference_s','sample_solver_s','download_s','call_s')]+[('stages_s',C.c_double*8)]+[(n,C.c_uint64) for n in ('persistent_bytes','fft_workspace_bytes','device_allocation_bytes')]
NAMES=('displace_y','displace_x','DPC_y','DPC_x','phase','transmission','match_score_neg_ssd')
class Library:
    def __init__(self,path):
        self.lib=C.CDLL(str(path));P=C.c_void_p;I=C.c_int;U=C.c_size_t
        declarations={'speckle_create':([C.POINTER(Config)],P),'speckle_destroy':([P],None),'speckle_error':([],C.c_char_p),
            'speckle_prepare':([P,P,U,C.POINTER(C.c_double)],I),'speckle_solve':([P,P,U,C.POINTER(P),C.POINTER(Timing)],I),
            'speckle_host_alloc':([U,I],P),'speckle_host_free':([P],I),'speckle_descriptor_bytes':([P,I],C.c_uint64),'speckle_copy_descriptor':([P,I,P,U],I)}
        for n,(args,result) in declarations.items():f=getattr(self.lib,n);f.argtypes=args;f.restype=result
    def check(self,code):
        if code:raise RuntimeError(self.lib.speckle_error().decode())
    def host(self,array,mapped=False):return Host(self,array,mapped)
class Host:
    def __init__(self,lib,array,mapped):
        self.lib=lib;self.pointer=lib.lib.speckle_host_alloc(array.nbytes,int(mapped))
        if not self.pointer:raise RuntimeError(lib.lib.speckle_error().decode())
        self.storage=(C.c_ubyte*array.nbytes).from_address(self.pointer);self.array=np.ctypeslib.as_array(self.storage).view(array.dtype).reshape(array.shape);np.copyto(self.array,array)
    def close(self):
        if self.pointer:self.lib.check(self.lib.lib.speckle_host_free(self.pointer));self.pointer=None
    def __enter__(self):return self.array
    def __exit__(self,*args):self.close()
class Session:
    def __init__(self,lib,shape,mode,dtype='uint16',low_memory=False,mapped_input=False):
        self.lib=lib;self.config=Config(*shape,int(mode=='wsvt'),int(dtype=='uint16'),int(low_memory),16,0,int(mapped_input),1.625e-6,.4845,12400.)
        self.handle=lib.lib.speckle_create(C.byref(self.config));self.outputs=None
        if not self.handle:raise RuntimeError(lib.lib.speckle_error().decode())
    def prepare_reference(self,a):
        t=C.c_double();self.lib.check(self.lib.lib.speckle_prepare(self.handle,a.ctypes.data,a.nbytes,C.byref(t)));return t.value
    def solve(self,a):
        h,w=self.config.height,self.config.width
        # New caller-owned arrays: later solves/destruction cannot overwrite past results.
        out={n:np.empty((h,w) if n=='transmission' else (h-20,w-20),np.float32) for n in NAMES};p=(C.c_void_p*7)(*[out[n].ctypes.data for n in NAMES]);t=Timing()
        self.lib.check(self.lib.lib.speckle_solve(self.handle,a.ctypes.data,a.nbytes,p,C.byref(t)))
        info={n:getattr(t,n) for n in ('reference_s','sample_solver_s','download_s','call_s','persistent_bytes','fft_workspace_bytes','device_allocation_bytes')};info['stages_s']=list(t.stages_s)
        return out,info
    def descriptor(self,index):
        size=self.lib.lib.speckle_descriptor_bytes(self.handle,index)
        if not size:return None
        a=np.empty(size//4,np.float32);self.lib.check(self.lib.lib.speckle_copy_descriptor(self.handle,index,a.ctypes.data,a.nbytes));return a
    def close(self):
        if self.handle:self.lib.lib.speckle_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
