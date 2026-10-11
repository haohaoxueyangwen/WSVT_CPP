"""Two-platform independent-process old/native comparison, fixed numerical protocol."""
from pathlib import Path
import argparse,csv,ctypes as C,gc,hashlib,importlib.util,json,os,platform,resource,statistics,subprocess,sys,time,traceback
import numpy as np
R=Path(__file__).resolve().parent;RELEASE=R/'release';IS_ORIN=platform.machine()=='aarch64';TARGET='jetson' if IS_ORIN else 'desktop'
sys.path.insert(0,str(RELEASE/'tools'));from native_api import Library,Session,NAMES
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def mark(folder,stage):write(folder/'stage.json',dict(stage=stage,time=time.time(),pid=os.getpid()))
def prepare():
    from validate_native import load
    folder=R/'inputs';folder.mkdir(exist_ok=False);manifest={}
    for mode in ('wsvt','wxst'):
        s,r=load(mode);manifest[mode]={}
        for n,a in zip(('sample','reference'),(s,r)):
            path=folder/(mode+'_'+n+'.npy');np.save(path,a,allow_pickle=False);manifest[mode][n]=dict(shape=list(a.shape),dtype=str(a.dtype),canonical_sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest(),file_sha256=sha(path),bytes=a.nbytes)
        if mode=='wsvt':
            assert manifest[mode]['sample']['canonical_sha256']=='80534d45fe6937e857dd4ea50682e2c565bed1d99f9b79eef0a9da34ba4940df'
            assert manifest[mode]['reference']['canonical_sha256']=='414af5ce3a38ee5c9aa0a7a15d8a91917b95134ba236998bd80177909a2e618d'
        del s,r;gc.collect()
    write(R/'input_manifest.json',manifest);print('INPUTS_PREPARED',flush=True)
def child(mode,profile):
    folder=R/mode/profile;folder.mkdir(parents=True,exist_ok=False);mark(folder,'load_inputs');t=time.perf_counter();s,r=[np.load(R/'inputs'/(mode+'_'+n+'.npy')) for n in ('sample','reference')];read_s=time.perf_counter()-t
    native=not profile.startswith('original');pinned=profile.endswith('_pinned') or '_mapped' in profile;mapped='_mapped' in profile;low='memory' in profile
    lib=Library(RELEASE/('build_'+TARGET)/'lib/libspeckle_native.so') if native or pinned else None;hosts=[]
    if pinned:
        for a in (s,r):hosts.append(lib.host(a,mapped))
        del s,r;s,r=[v.array for v in hosts];gc.collect()
    mark(folder,'initialize');t=time.perf_counter()
    if native:q=Session(lib,s.shape[-2:],mode,str(s.dtype),low,mapped)
    else:
        import torch
        torch.set_num_threads(2);spec=importlib.util.spec_from_file_location('base',R/'baseline/speckle_edge/runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        q=m.EdgeSession(s.shape[-2:],mode,'baseline' if mode=='wsvt' else 'fused',spline='gpu',input_dtype=str(s.dtype),cpu_threads=2)
    constructor_s=time.perf_counter()-t
    def run():
        before=os.getloadavg();t=time.perf_counter();reference=q.prepare_reference(r);out,info=q.solve(s);api=time.perf_counter()-t
        sample=info['sample_solver_s'] if native else info['solver_s'];download=info['download_s'] if native else info['call_with_download_s']-info['solver_s']
        row=dict(solver_s=reference+sample,api_s=api,reference_s=reference,sample_s=sample,download_s=download,detail=info,load_before=before)
        assert all(np.isfinite(a).all() for a in out.values());return out,row
    mark(folder,'warmup');out,warm=run();del out;gc.collect()
    if not native:torch.cuda.reset_peak_memory_stats()
    mark(folder,'formal');rows=[]
    for j in range(5):
        out,row=run();del out;row['repeat']=j;rows.append(row);write(folder/'records.json',rows);print('FORMAL',mode,profile,j,row['solver_s'],row['api_s'],flush=True)
    memory=dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        rss_scope='Lifetime through end of measured solves, includes input read/imports/initialization/warmup; subsequent verification and saving excluded.')
    if native:memory.update(native_explicit_device_bytes=rows[0]['detail']['device_allocation_bytes'],native_persistent_bytes=rows[0]['detail']['persistent_bytes'],native_fft_workspace_bytes=rows[0]['detail']['fft_workspace_bytes'])
    else:memory.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),persistent_device_bytes=rows[0]['detail']['persistent_device_bytes'])
    mark(folder,'verify');features={}
    for j,n in enumerate(('ref_fine','ref_coarse','sample_fine','sample_coarse')):
        if native:
            a=q.descriptor(j)
            if a is not None:features[n]=hashlib.sha256(memoryview(a).cast('B')).hexdigest()
            del a
        elif n in q.b:
            hh=hashlib.sha256()
            for jj in range(0,q.b[n].shape[0],64):hh.update(q.b[n][jj:jj+64].cpu().numpy().tobytes())
            features[n]=hh.hexdigest()
    out,_=run();mark(folder,'save_outputs');t=time.perf_counter()
    for n,a in out.items():np.save(folder/(n+'.npy'),a,allow_pickle=False)
    save_s=time.perf_counter()-t
    summary={k:dict(mean=statistics.mean(v[k] for v in rows),sample_sd=statistics.stdev(v[k] for v in rows)) for k in ('solver_s','api_s','reference_s','sample_s','download_s')}
    report=dict(mode=mode,profile=profile,platform=TARGET,shape=list(s.shape),input_dtype=str(s.dtype),transport='mapped pinned host' if mapped else 'pinned host' if pinned else 'pageable host',
        summary=summary,memory=memory,warmup=warm,records=rows,features_sha256=features,read_s=read_s,constructor_s=constructor_s,save_s=save_s,
        geometry=dict(pixel_m=1.625e-6,distance_m=.4845,energy_ev=12400.),settings=dict(half=10,coarse=5,fine=4,levels=2,iterations=1,cut=1,template=0 if mode=='wsvt' else 7,frames=100 if mode=='wsvt' else 1,host_threads=2,spline='gpu',reference_policy='each-pair',warmup=1,repeats=5,low_memory=low,chunk=16),
        scope='Pair solver includes reference rebuild and sample compute, final synchronization, no outputs. API ready host arrays to seven new independently owned NumPy outputs. Input read, pinned/mapped host setup, constructor, file saving excluded and recorded; no Python/PyTorch reconstruction in native profiles.',
        hashes=dict(driver=sha(__file__),library=sha(RELEASE/('build_'+TARGET)/'lib/libspeckle_native.so') if native else sha(R/'baseline/speckle_edge/_native/libedge_kernels.so')))
    with (folder/'timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['repeat','solver_s','api_s','reference_s','sample_s','download_s']);w.writeheader();w.writerows({k:v[k] for k in w.fieldnames} for v in rows)
    write(folder/'report.json',report);q.close()
    for v in hosts:v.close()
    mark(folder,'complete');print('CHILD_COMPLETE',mode,profile,flush=True)
class NVML:
    class Process(C.Structure):_fields_=[('pid',C.c_uint),('usedGpuMemory',C.c_ulonglong),('gpuInstanceId',C.c_uint),('computeInstanceId',C.c_uint)]
    def __init__(self):
        self.lib=C.CDLL('libnvidia-ml.so.1');assert self.lib.nvmlInit_v2()==0;self.handle=C.c_void_p();assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,C.byref(self.handle))==0
        self.lib.nvmlDeviceGetComputeRunningProcesses_v2.argtypes=[C.c_void_p,C.POINTER(C.c_uint),C.POINTER(self.Process)]
    def processes(self):
        n=C.c_uint(64);v=(self.Process*64)();code=self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,C.byref(n),v)
        if code:raise RuntimeError('NVML process query failed '+str(code))
        return {p.pid:p.usedGpuMemory for p in v[:n.value]}
def sample(pid,nvml):
    d=dict(time=time.time(),load=os.getloadavg())
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1)
        if k in ('MemTotal','MemAvailable','SwapFree'):d[k]=int(v.strip().split()[0])*1024
    try:
        for line in Path('/proc/%d/status'%pid).read_text().splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):k,v=line.split(':',1);d[k]=int(v.strip().split()[0])*1024
    except FileNotFoundError:pass
    if nvml:
        procs=nvml.processes();d['nvml_process_bytes']=procs.get(pid,0);d['gpu_processes']=procs
        others=[p for p in procs if p!=pid]
        if others:raise RuntimeError('Competing GPU process, stop benchmark: '+str(others))
    return d
def monitored(command,folder,tag,nvml):
    logfile=open(R/(tag+'.log'),'w');p=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT);samples=[]
    tg=None;tf=None
    if IS_ORIN:tf=open(R/(tag+'_tegrastats.log'),'w');tg=subprocess.Popen(['tegrastats','--interval','200'],stdout=tf,stderr=subprocess.STDOUT)
    try:
        while p.poll() is None:
            v=sample(p.pid,nvml)
            try:v['stage']=json.loads((folder/'stage.json').read_text())['stage']
            except (FileNotFoundError,json.JSONDecodeError):v['stage']='cli' if 'cli_' in tag else 'start'
            samples.append(v);time.sleep(.025)
    finally:
        if tg:tg.terminate();tg.wait();tf.close()
        logfile.close()
    write(R/(tag+'_memory.json'),samples)
    if p.returncode:raise RuntimeError('Failed '+tag+'; see log')
    report=json.loads((folder/'report.json').read_text());formal=[v for v in samples if v['stage'] in ('formal','cli')];assert formal
    report['sampled_memory']=dict(peak_rss_bytes=max(v.get('VmRSS',0) for v in formal),system_used_peak_bytes=max(v['MemTotal']-v['MemAvailable'] for v in formal),
        nvml_process_peak_bytes=max(v.get('nvml_process_bytes',0) for v in formal) if nvml else None,
        sample_interval_s=.025,scope='25ms sampled peaks are lower bounds. Adapter profiles: formal only. CLI profiles: whole lifetime including constructor. System RAM includes background/cache. Unified DRAM metrics overlap; do not sum.')
    write(folder/'report.json',report);return report
def batch(mode):
    assert 'ALL_SMOKE_PASS' in (R/'smoke.log').read_text(),'Run smoke before benchmark'
    if os.getloadavg()[0]>(4 if IS_ORIN else 8):raise RuntimeError('CPU load unsuitable')
    nvml=None if IS_ORIN else NVML()
    if nvml and nvml.processes():raise RuntimeError('GPU occupied before benchmark')
    profiles=['original','native_speed','native_memory','original_pinned','native_speed_pinned'] if mode=='wsvt' else ['original','native_speed','original_pinned','native_speed_pinned']
    if IS_ORIN:profiles+=['native_speed_mapped']+(['native_memory_mapped'] if mode=='wsvt' else [])
    reports={}
    for profile in profiles:
        write(R/(mode+'_status.json'),dict(state='running',profile=profile));folder=R/mode/profile
        reports[profile]=monitored([sys.executable,'-u',__file__,'child','--mode',mode,'--profile',profile],folder,mode+'_'+profile,nvml);write(R/(mode+'_summary.json'),reports);print('PROFILE_COMPLETE',mode,profile,flush=True);time.sleep(3)
    # Actual standalone C++ executable; pinned inputs/outputs and file I/O scoped separately.
    for workspace in (('speed','low-memory') if mode=='wsvt' else ('speed',)):
        policies=('copy','mapped') if IS_ORIN else ('copy',)
        for policy in policies:
            tag='cli_'+mode+'_'+workspace.replace('-','_')+'_'+policy;folder=R/mode/tag
            command=[str(RELEASE/('build_'+TARGET)/'bin/speckle_gpu'),'--mode',mode,'--sample',str(R/'inputs'/(mode+'_sample.npy')),'--reference',str(R/'inputs'/(mode+'_reference.npy')),'--out',str(folder),'--platform',TARGET,'--workspace',workspace,'--input-policy',policy,'--pixel-m','1.625e-6','--distance-m','.4845','--energy-ev','12400.','--warmup','1','--repeats','5']
            write(R/(mode+'_status.json'),dict(state='running',profile=tag));reports[tag]=monitored(command,folder,tag,nvml);write(R/(mode+'_summary.json'),reports);print('CLI_COMPLETE',tag,flush=True);time.sleep(3)
    write(R/(mode+'_status.json'),dict(state='complete'));print('MODE_COMPLETE',mode,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['prepare','batch','child']);ap.add_argument('--mode',choices=['wsvt','wxst']);ap.add_argument('--profile');args=ap.parse_args()
    try:
        if args.task=='prepare':prepare()
        elif args.task=='batch':batch(args.mode)
        else:child(args.mode,args.profile)
    except Exception as e:
        if args.task=='batch':write(R/(args.mode+'_status.json'),dict(state='failed',error=str(e)))
        traceback.print_exc();sys.exit(1)
