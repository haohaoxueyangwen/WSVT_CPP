"""Bounded old/new GPU benchmark; real BB11 inputs; no CPU/subpixel changes."""
import argparse,csv,gc,hashlib,importlib.util,json,os,statistics,subprocess,time
from pathlib import Path
import numpy as np
import tifffile
import torch
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'edge_gpu_workstation_20261011/package'
NEW=ROOT/'package'
DATA=ROOT.parents[1]/'wsvtdata/BB11_20231102_20261002/raw'
NAMES=('displace_y','displace_x','DPC_y','DPC_x','phase','transmission','match_score_neg_ssd')
def write(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def cmd(args):return subprocess.check_output(args,text=True).strip()
def state():
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),load=os.getloadavg(),
        gpu=cmd(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw','--format=csv,noheader']),
        processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']))
def idle():
    s=state()
    if any(line and int(line.split(',')[0])!=os.getpid() for line in s['processes'].splitlines()):raise RuntimeError('Competing CUDA process: '+s['processes'])
    if s['load'][0]>8:raise RuntimeError('CPU load unsuitable for timing')
    return s
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path/'speckle_edge/runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
BASE=module(OLD,'old_runtime');CAND=module(NEW,'new_runtime')
def session(mode,shape,variant,dtype='uint16',chunk=32):
    cls=(BASE if variant=='original' else CAND).EdgeSession
    kwargs={} if variant=='original' else dict(wsvt_workspace='materialized' if variant=='copy_only' else 'streamed',spatial_chunk=chunk,numpy_output_copy=False)
    return cls(shape,mode,'baseline' if mode=='wsvt' else 'fused',input_dtype=dtype if mode=='wsvt' else 'float32',**kwargs)
def equal_outputs(a,b):
    report={n:dict(exact=bool(np.array_equal(a[n],b[n])),max_abs=float(np.max(np.abs(a[n].astype(np.float64)-b[n]))),sha256=hashlib.sha256(a[n].tobytes()).hexdigest()) for n in NAMES}
    if not all(v['exact'] for v in report.values()):raise RuntimeError('Output mismatch: '+json.dumps(report))
    return report
def equal_features(a,b):
    r={k:bool(torch.equal(a.b[k],b.b[k])) for k in ('ref_fine','ref_coarse','sample_fine','sample_coarse') if k in a.b and k in b.b}
    if not all(r.values()):raise RuntimeError('Descriptor mismatch: '+str(r))
    return r
def load(mode):
    out=[];manifest=[];start=time.perf_counter()
    for tag in ('sample','ref'):
        paths=[DATA/(tag+'_1_%05d.tif'%j) for j in range(1,101 if mode=='wsvt' else 2)]
        a=np.empty((len(paths),2048,2048),np.uint16)
        for j,p in enumerate(paths):a[j]=tifffile.imread(p);manifest.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
        out.append(a if mode=='wsvt' else a[0].astype(np.float32))
    frozen=json.loads((ROOT.parent/'edge_gpu_workstation_20261011'/('full_'+mode)/'report.json').read_text())['input_manifest']
    expected={v['path']:v['sha256'] for v in frozen}
    assert all(expected.get(v['path'])==v['sha256'] for v in manifest),'Inputs changed'
    return out,manifest,time.perf_counter()-start
def smoke():
    s=np.stack([tifffile.imread(DATA/('sample_1_%05d.tif'%j))[896:1153,896:1159] for j in range(1,101)])
    r=np.stack([tifffile.imread(DATA/('ref_1_%05d.tif'%j))[896:1153,896:1159] for j in range(1,101)])
    results=[]
    for mode,dtype in [('wsvt','uint16'),('wsvt','float32'),('wxst','float32')]:
        raw_s=np.ascontiguousarray(s if mode=='wsvt' else s[0],dtype=dtype);raw_r=np.ascontiguousarray(r if mode=='wsvt' else r[0],dtype=dtype)
        for chunk in (16,32,100):
            with session(mode,raw_s.shape[-2:],'original',dtype) as a,session(mode,raw_s.shape[-2:],'streamed',dtype,chunk) as b:
                a.prepare_reference(raw_r);b.prepare_reference(raw_r);ao,_=a.solve(raw_s);bo,_=b.solve(raw_s)
                checks=equal_outputs(ao,bo);features=equal_features(a,b);held={k:v.copy() for k,v in bo.items()}
                bo['phase'][0,0]+=1;held['phase'][0,0]+=1
                bo2,_=b.solve(raw_r)
                assert all(np.array_equal(bo[k],held[k]) and not np.shares_memory(bo[k],bo2[k]) for k in NAMES)
                b.close();gc.collect();assert all(np.array_equal(bo[k],held[k]) for k in NAMES)
                results.append(dict(mode=mode,dtype=dtype,shape=raw_s.shape,chunk=chunk,outputs=checks,features=features,ownership_pass=True))
                print('smoke pass',mode,dtype,chunk,flush=True)
    # Zero standard deviation is a branch-sensitive engineering check, not a truth experiment.
    constant=np.full((100,33,37),123,np.uint16)
    with session('wsvt',(33,37),'original') as a,session('wsvt',(33,37),'streamed',chunk=16) as b:
        for q in (a,b):q.prepare_reference(constant)
        ao,_=a.solve(constant);bo,_=b.solve(constant)
        results.append(dict(constant=equal_outputs(ao,bo),features=equal_features(a,b)))
    write(ROOT/'smoke.json',results)
def run_one(q,s,r):
    before=idle();torch.cuda.synchronize();t=time.perf_counter()
    ref=q.prepare_reference(r);out,info=q.solve(s);api=time.perf_counter()-t
    for k,v in out.items():assert np.isfinite(v).all(),k
    row=dict(solver_s=ref+info['solver_s'],api_s=api,reference_s=ref,sample_s=info['solver_s'],download_s=info['call_with_download_s']-info['solver_s'],detail=info,load_before=before)
    return out,row
def full(mode):
    folder=ROOT/('controlled_'+mode);folder.mkdir(exist_ok=False)
    write(folder/'status.json',dict(state='running'))
    (s,r),manifest,read_s=load(mode);chunks=[];best=16
    if False:
        for chunk in (16,32,100):
            with session(mode,(2048,2048),'streamed',chunk=chunk) as q:
                run_one(q,s,r)
                rows=[]
                for rep in range(3):
                    out,row=run_one(q,s,r);row.update(chunk=chunk,repeat=rep);rows.append(row);del out
                chunks.extend(rows);print('chunk',chunk,'solver',statistics.mean(x['solver_s'] for x in rows),'resident',rows[0]['detail']['persistent_device_bytes'],flush=True)
        write(folder/'chunk_trials.json',chunks)
        best=min((16,32,100),key=lambda c:statistics.mean(x['solver_s'] for x in chunks if x['chunk']==c))
    qs={v:session(mode,(2048,2048),v,chunk=best) for v in ('original','copy_only','streamed')}
    rows=[];warm={};last={};checks={}
    try:
        for v,q in qs.items():
            out,row=run_one(q,s,r);warm[v]=row;last[v]=out;print('warmup',mode,v,row['solver_s'],flush=True)
        for v in ('copy_only','streamed'):
            checks[v]=dict(outputs=equal_outputs(last['original'],last[v]),features=equal_features(qs['original'],qs[v]))
        for v in ('original','streamed'):np.savez_compressed(folder/(v+'_outputs.npz'),**last[v])
        last.clear();del out;gc.collect()
        for rep in range(5):
            order=list(qs);order=order[rep%3:]+order[:rep%3]
            for v in order:
                out,row=run_one(qs[v],s,r);row.update(variant=v,repeat=rep);rows.append(row)
                write(folder/'records.json',rows);print('formal',mode,rep,v,row['solver_s'],row['api_s'],flush=True)
                del out
        # Full outputs and feature parity were checked before formal timing.
        summary={}
        for v in qs:
            rr=[x for x in rows if x['variant']==v];summary[v]={}
            for k in ('solver_s','api_s','reference_s','sample_s','download_s'):
                values=[x[k] for x in rr];summary[v][k]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
            summary[v]['persistent_device_bytes']=rr[0]['detail']['persistent_device_bytes']
            summary[v]['stages_s']={k:statistics.mean(x['detail']['gpu_event_s'][k] for x in rr) for k in rr[0]['detail']['gpu_event_s']}
        report=dict(mode=mode,shape=list(s.shape),selected_chunk=best,chunks_exploratory=chunks,warmup=1,repeats=5,summary=summary,checks=checks,records=rows,warmup_records=warm,
            input_manifest=manifest,read_and_hash_s=read_s,settings=dict(half=10,coarse=5,fine=4,levels=2,iterations=1,cut=1,template=7 if mode=='wxst' else 0,frames=1 if mode=='wxst' else 100,spline='gpu',input_dtype=str(s.dtype),host_threads=2,device=0),
            timing_scope={'solver':'reference rebuilt + sample solver, including H2D and final GPU sync; excludes output download, input validation, initialization and disk I/O',
                         'api':'ready contiguous host arrays -> seven independent writable NumPy outputs, including validation and reference rebuild; excludes initialization and disk I/O',
                         'memory':'persistent GPU buffers only; not process/allocator peak; simultaneous baseline/candidate sessions are resident for alternating comparison'},
            hardware=dict(host=cmd(['hostname']),cpu=cmd(['lscpu']),gpu=cmd(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version,power.limit','--format=csv']),torch=torch.__version__,cuda_native='12.8',arch=89),
            hashes=dict(benchmark=sha(__file__),old_library=sha(OLD/'speckle_edge/_native/libedge_kernels.so'),new_library=sha(NEW/'speckle_edge/_native/libedge_kernels.so'),new_runtime=sha(NEW/'speckle_edge/runtime.py')),load_end=state())
        with (folder/'timing.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['variant','repeat','solver_s','api_s','reference_s','sample_s','download_s']);w.writeheader()
            for row in rows:w.writerow({k:row[k] for k in w.fieldnames})
        write(folder/'report.json',report);write(folder/'status.json',dict(state='complete'));print(json.dumps(summary,indent=2),flush=True)
    finally:
        for q in qs.values():q.close()
def main():
    torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('task',choices=['smoke','wxst','wsvt']);a=ap.parse_args();idle()
    if a.task=='smoke':smoke()
    else:full(a.task)
if __name__=='__main__':main()
