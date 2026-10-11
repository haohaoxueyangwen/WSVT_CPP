"""Fixed engineering comparisons against preserved GPU0.1.0, not physical truth."""
from pathlib import Path
import gc,hashlib,importlib.util,json,platform,shutil,sys
import numpy as np
import torch
R=Path(__file__).resolve().parent;RELEASE=R/'release';IS_ORIN=platform.machine()=='aarch64';TARGET='jetson' if IS_ORIN else 'desktop'
sys.path.insert(0,str(RELEASE/'tools'));from native_api import Library,Session,NAMES
OLD=(R.parent/'memory_opt_20261011/baseline') if IS_ORIN else (R.parent/'edge_gpu_workstation_20261011/package')
if not (R/'baseline').exists():shutil.copytree(OLD,R/'baseline',ignore=shutil.ignore_patterns('build','dist','__pycache__'))
spec=importlib.util.spec_from_file_location('baseline_runtime',R/'baseline/speckle_edge/runtime.py');BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def load(mode,crop=False):
    if IS_ORIN:
        with np.load(R.parent/'edge_adaptation_20261011/wsvt100_full2048.npz') as z:
            s,r=z['sample'],z['reference']
        if mode=='wxst':s,r=s[0].astype(np.float32),r[0].astype(np.float32)
    else:
        import tifffile
        data=R.parents[1]/'wsvtdata/BB11_20231102_20261002/raw';values=[]
        for tag in ('sample','ref'):
            a=np.empty((100 if mode=='wsvt' else 1,2048,2048),np.uint16)
            for j in range(a.shape[0]):a[j]=tifffile.imread(data/(tag+'_1_%05d.tif'%(j+1)))
            values.append(a if mode=='wsvt' else a[0].astype(np.float32))
        s,r=values
    if crop:return [np.ascontiguousarray(a[...,896:1153,896:1159]) for a in (s,r)]
    return s,r
def original(mode,shape,dtype):return BASE.EdgeSession(shape,mode,'baseline' if mode=='wsvt' else 'fused',spline='gpu',input_dtype=dtype,cpu_threads=2)
def compare(expected,actual):
    result={}
    for n in NAMES:
        a=expected[n].astype(np.float64);b=actual[n].astype(np.float64);d=b-a
        v=dict(exact=bool(np.array_equal(expected[n],actual[n])),max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))),finite=bool(np.isfinite(b).all()))
        if n=='phase':
            v['max_abs_limit']=2e-3+1e-5*float(np.max(np.abs(a)));v['rmse_limit']=2e-4+1e-6*float(np.sqrt(np.mean(a*a)));v['pass']=v['finite'] and v['max_abs']<=v['max_abs_limit'] and v['rmse']<=v['rmse_limit']
        else:v['pass']=v['finite'] and v['exact']
        result[n]=v
    return result
def smoke():
    torch.set_num_threads(2);lib=Library(RELEASE/('build_'+TARGET)/'lib/libspeckle_native.so');s,r=load('wsvt',True);results=[]
    cases=[('wxst_odd','wxst',s[0].astype(np.float32),r[0].astype(np.float32)),('wsvt_odd','wsvt',s,r),('wsvt_even','wsvt',np.ascontiguousarray(s[:,:256,:258]),np.ascontiguousarray(r[:,:256,:258])),('wsvt_float','wsvt',s.astype(np.float32),r.astype(np.float32)),('constant','wsvt',np.full((100,33,37),123,np.uint16),np.full((100,33,37),123,np.uint16))]
    for label,mode,a,b in cases:
        with original(mode,a.shape[-2:],str(a.dtype)) as old:
            old.prepare_reference(b);expected,_=old.solve(a)
            options=[(False,False),(True,False)] if mode=='wsvt' else [(False,False)]
            if IS_ORIN:options += [(False,True)]+([(True,True)] if mode=='wsvt' else [])
            for low,mapped in options:
                ha=lib.host(a,mapped) if mapped else None;hb=lib.host(b,mapped) if mapped else None
                try:
                    aa,bb=(ha.array,hb.array) if mapped else (a,b)
                    with Session(lib,a.shape[-2:],mode,str(a.dtype),low,mapped) as q:
                        q.prepare_reference(bb);actual,_=q.solve(aa);checks=compare(expected,actual);features={}
                        for j,n in enumerate(('ref_fine','ref_coarse','sample_fine','sample_coarse')):
                            v=q.descriptor(j)
                            if v is not None:features[n]=bool(np.array_equal(old.b[n].cpu().numpy().ravel(),v))
                        record=dict(case=label,low_memory=low,mapped_input=mapped,fields=checks,descriptors=features);results.append(record);write(R/'smoke.json',results)
                        assert all(v['pass'] for v in checks.values()) and all(features.values()),record
                        held={n:v.copy() for n,v in actual.items()};other,_=q.solve(bb)
                        assert all(np.array_equal(actual[n],held[n]) and not np.shares_memory(actual[n],other[n]) for n in NAMES)
                        q.close();gc.collect();assert all(np.array_equal(actual[n],held[n]) for n in NAMES);record['ownership_pass']=True
                        print('SMOKE_PASS',label,'low',low,'mapped',mapped,'phase',checks['phase'],flush=True)
                finally:
                    if ha:ha.close();hb.close()
    write(R/'smoke.json',results);print('ALL_SMOKE_PASS',flush=True)
if __name__=='__main__':smoke()
