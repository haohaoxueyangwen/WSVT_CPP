"""Edge-case validation, including API rejection and changed input re-validation."""
from pathlib import Path
import ctypes as C, json, sys
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
from native_api import Library,Session
lib=Library(root/'build_gpu/lib/libspeckle_native.so')
fn=lib.lib.speckle_all_finite;fn.argtypes=[C.c_void_p,C.c_size_t];fn.restype=C.c_bool
cases=0
for length in (0,1,15,16,17,31,32,33,127,128,129,4097):
    a=np.zeros(length+1,np.float32)[1:] # Deliberately unaligned view.
    if length:
        a[:]=np.resize(np.array([0.,-0.,np.finfo(np.float32).max,np.nextafter(np.float32(0),np.float32(1))],np.float32),length)
    assert fn(a.ctypes.data,length);cases+=1
    for position in set((0,length//2,length-1)) if length else ():
        saved=a[position]
        for bad in (np.nan,np.inf,-np.inf):
            a[position]=bad;assert not fn(a.ctypes.data,length);cases+=1
        a[position]=saved
with Session(lib,(128,128),'wxst','float32') as q:
    a=np.ones((128,128),np.float32)
    for stage in ('reference','sample'):
        if stage=='sample':q.prepare_reference(a)
        for pos in (0,a.size//2,a.size-1):
            for bad in (np.nan,np.inf,-np.inf):
                a.flat[pos]=bad
                try:
                    q.prepare_reference(a) if stage=='reference' else q.solve(a)
                except RuntimeError as e:
                    assert 'Nonfinite' in str(e)
                else:raise AssertionError('Nonfinite API input accepted')
                finally:a.flat[pos]=1
                cases+=1
report=dict(pass_gate=True,cases=cases,host_validation_threads=1,
    checks=['IEEE finite extremes/subnormals/signed zero','unaligned and SIMD tails','NaN and +/-Inf','revalidate changed reference/sample buffers'])
(root/'evidence/finite_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
