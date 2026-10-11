"""Frozen BB11 WSVT benchmark. Independent process/context for each profile."""
from pathlib import Path
import argparse,csv,gc,hashlib,importlib.util,json,os,resource,statistics,subprocess,sys,time,traceback
R=Path(__file__).resolve().parent
DATA=R.parent/'edge_adaptation_20261011/wsvt100_full2048.npz'
PROFILES=('original','speed','low_memory')
FIELDS=('displace_y','displace_x','DPC_y','DPC_x','phase','transmission','match_score_neg_ssd')
def write(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def mark(folder,stage):write(folder/'stage.json',dict(stage=stage,pid=os.getpid(),time=time.time()))
def module(profile):
    path=R/('baseline' if profile=='original' else 'package')/'speckle_edge/runtime.py'
    spec=importlib.util.spec_from_file_location('runtime_'+profile,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def session(profile,shape,dtype='uint16'):
    m=module(profile);kw={} if profile=='original' else dict(wsvt_workspace='materialized' if profile=='speed' else 'streamed',spatial_chunk=16,numpy_output_copy=False)
    return m.EdgeSession(shape,'wsvt','baseline',spline='gpu',half=10,pixel=1.625e-6,distance=.4845,energy=12400.,cpu_threads=2,input_dtype=dtype,**kw)
def load():
    import numpy as np
    with np.load(DATA,allow_pickle=False) as z:s,r=z['sample'],z['reference']
    assert s.shape==r.shape==(100,2048,2048) and s.dtype==r.dtype==np.uint16
    return s,r
def smoke():
    import numpy as np,torch
    torch.set_num_threads(2);s,r=load()
    crops=(np.ascontiguousarray(s[:,896:1153,896:1159]),np.ascontiguousarray(r[:,896:1153,896:1159]));del s,r;gc.collect()
    results=[]
    for label,dtype,h,w in [('odd_u16','uint16',257,263),('even_u16','uint16',256,258),('odd_float','float32',257,263),('constant','uint16',33,37)]:
        a,b=[np.full((100,h,w),123,np.uint16) for _ in range(2)] if label=='constant' else [np.ascontiguousarray(v[:,:h,:w],dtype=dtype) for v in crops]
        with session('original',(h,w),dtype) as old:
            old.prepare_reference(b);expected,_=old.solve(a)
            for profile in ('speed','low_memory'):
                with session(profile,(h,w),dtype) as q:
                    q.prepare_reference(b);actual,_=q.solve(a)
                    checks={n:bool(np.array_equal(expected[n],actual[n])) for n in FIELDS}
                    features={n:bool(torch.equal(old.b[n],q.b[n])) for n in ('ref_fine','ref_coarse','sample_fine','sample_coarse')}
                    assert all(checks.values()) and all(features.values()),(label,profile,checks,features)
                    held={n:v.copy() for n,v in actual.items()};actual['phase'][0,0]+=1;held['phase'][0,0]+=1
                    other,_=q.solve(b)
                    assert all(np.array_equal(actual[n],held[n]) and not np.shares_memory(actual[n],other[n]) for n in FIELDS)
                    q.close();gc.collect();assert all(np.array_equal(actual[n],held[n]) for n in FIELDS)
                    results.append(dict(case=label,profile=profile,outputs=checks,features=features,ownership_pass=True))
                    print('SMOKE_PASS',label,profile,flush=True)
    write(R/'smoke.json',results)
def child(profile):
    folder=R/profile;folder.mkdir(exist_ok=False);mark(folder,'import')
    import numpy as np,torch
    torch.set_num_threads(2);mark(folder,'load_inputs');start=time.perf_counter();s,r=load();read_s=time.perf_counter()-start
    mark(folder,'initialize');q=session(profile,(2048,2048));torch.cuda.synchronize()
    def run():
        torch.cuda.synchronize();t=time.perf_counter();reference=q.prepare_reference(r);out,info=q.solve(s);api=time.perf_counter()-t
        row=dict(solver_s=reference+info['solver_s'],api_s=api,reference_s=reference,sample_s=info['solver_s'],download_s=info['call_with_download_s']-info['solver_s'],detail=info,load=os.getloadavg())
        for n,a in out.items():assert np.isfinite(a).all(),n
        return out,row
    mark(folder,'warmup');out,warm=run();del out;gc.collect();torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();mark(folder,'formal');rows=[]
    for j in range(5):
        out,row=run();row['repeat']=j;rows.append(row);del out
        write(folder/'records.json',rows);print('FORMAL',profile,j,row['solver_s'],row['api_s'],flush=True)
    torch.cuda.synchronize()
    memory=dict(persistent_device_bytes=rows[0]['detail']['persistent_device_bytes'],
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_current_allocated_bytes=torch.cuda.memory_allocated(),cuda_current_reserved_bytes=torch.cuda.memory_reserved(),
        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='Allocator peaks reset after warmup; includes initialized session and all five pair solves. RSS peak through end of formal solves includes imports, input decoding, initialization and warmup, excludes subsequent output saving/descriptor verification. Shared DRAM metrics overlap; do not add.')
    mark(folder,'verify');features={}
    for n in ('ref_fine','ref_coarse','sample_fine','sample_coarse'):
        h=hashlib.sha256()
        for j in range(0,q.b[n].shape[0],64):h.update(q.b[n][j:j+64].cpu().numpy().tobytes())
        features[n]=h.hexdigest()
    out,_=run();mark(folder,'save_outputs');save_start=time.perf_counter()
    for n,a in out.items():np.save(folder/(n+'.npy'),a,allow_pickle=False)
    output_save_s=time.perf_counter()-save_start
    summary={k:dict(mean=statistics.mean(v[k] for v in rows),sample_sd=statistics.stdev(v[k] for v in rows)) for k in ('solver_s','api_s','reference_s','sample_s','download_s')}
    package=R/('baseline' if profile=='original' else 'package')
    report=dict(profile=profile,summary=summary,memory=memory,warmup=warm,repeats=5,records=rows,features_sha256=features,
        read_decode_s=read_s,output_save_s=output_save_s,shape=list(s.shape),input_dtype=str(s.dtype),
        settings=dict(half=10,coarse=5,fine=4,levels=2,iterations=1,cut=1,spatial_template=0,frames=100,spline='gpu',host_threads=2,device=0,wsvt_workspace='streamed' if profile=='low_memory' else 'materialized',spatial_chunk=16 if profile=='low_memory' else 100,numpy_output_copy=profile=='original',reference_policy='each-pair'),
        timing_scope=dict(solver='reference rebuild + sample solver, including input uploads and final GPU synchronization; excludes validation, output download, initialization and file I/O',api='ready contiguous host arrays -> seven independent writable NumPy arrays, including validation, reference rebuild and output download; excludes initialization and file I/O'),
        hashes=dict(runtime=sha(package/'speckle_edge/runtime.py'),native=sha(package/'speckle_edge/_native/libedge_kernels.so'),script=sha(__file__)),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device_name=torch.cuda.get_device_name(),properties=str(torch.cuda.get_device_properties(0))))
    write(folder/'report.json',report)
    with (folder/'timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['repeat','solver_s','api_s','reference_s','sample_s','download_s']);w.writeheader();w.writerows({k:v[k] for k in w.fieldnames} for v in rows)
    q.close();mark(folder,'complete');print('COMPLETE',profile,json.dumps(summary),flush=True)
def snapshot(pid):
    d={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1)
        if k in ('MemTotal','MemAvailable','MemFree','SwapTotal','SwapFree'):d[k]=int(v.strip().split()[0])*1024
    d.update(time=time.time(),load=os.getloadavg())
    try:
        for line in Path('/proc/%d/status'%pid).read_text().splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):k,v=line.split(':',1);d[k]=int(v.strip().split()[0])*1024
    except FileNotFoundError:pass
    return d
def batch():
    write(R/'status.json',dict(state='running',phase='smoke'))
    subprocess.run([sys.executable,'-u',__file__,'smoke'],check=True)
    all_reports={}
    for profile in PROFILES:
        write(R/'status.json',dict(state='running',phase=profile))
        log=open(R/(profile+'.log'),'w');p=subprocess.Popen([sys.executable,'-u',__file__,'child','--profile',profile],stdout=log,stderr=subprocess.STDOUT)
        folder=R/profile;samples=[];baseline=snapshot(p.pid)
        tf=open(R/(profile+'_tegrastats.log'),'w');tg=subprocess.Popen(['tegrastats','--interval','200'],stdout=tf,stderr=subprocess.STDOUT)
        try:
            while p.poll() is None:
                sample=snapshot(p.pid)
                try:sample['stage']=json.loads((folder/'stage.json').read_text())['stage']
                except (FileNotFoundError,json.JSONDecodeError):sample['stage']='start'
                samples.append(sample);time.sleep(.025)
        finally:tg.terminate();tg.wait();tf.close();log.close()
        write(R/(profile+'_memory_samples.json'),dict(baseline=baseline,samples=samples,sample_interval_s=.025))
        if p.returncode:raise RuntimeError('%s failed, see log'%profile)
        report=json.loads((folder/'report.json').read_text());formal=[s for s in samples if s['stage']=='formal']
        assert formal,'No formal memory samples'
        report['memory'].update(sampled_formal_peak_rss_bytes=max(s.get('VmRSS',0) for s in formal),
            sampled_formal_system_used_bytes=max(s['MemTotal']-s['MemAvailable'] for s in formal),
            system_used_before_process_bytes=baseline['MemTotal']-baseline['MemAvailable'],
            sampled_formal_system_used_delta_bytes=max(s['MemTotal']-s['MemAvailable'] for s in formal)-(baseline['MemTotal']-baseline['MemAvailable']),
            sampled_note='25 ms nominal sampler, lower bounds on transient peaks; system metric includes other processes/cache and is not GPU VRAM. tegrastats 200 ms recorded separately.')
        write(folder/'report.json',report);all_reports[profile]=report
        write(R/'summary.json',all_reports);print('PROFILE_COMPLETE',profile,flush=True);time.sleep(3)
    write(R/'status.json',dict(state='complete'));print('BATCH_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['batch','smoke','child']);ap.add_argument('--profile',choices=PROFILES);args=ap.parse_args()
    try:
        if args.task=='batch':batch()
        elif args.task=='smoke':smoke()
        else:child(args.profile)
    except Exception as e:
        if args.task=='batch':write(R/'status.json',dict(state='failed',error=str(e)))
        traceback.print_exc();sys.exit(1)
