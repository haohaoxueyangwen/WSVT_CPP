"""Authorized serial matched-array benchmark. No algorithm or dataset edits."""
from pathlib import Path
import argparse, csv, ctypes as C, gc, hashlib, importlib, importlib.util
import json, os, statistics, subprocess, sys, time, traceback
import numpy as np

P = Path('/home/yrc/vscodespace/research/WSVT_project')
E = P / 'bench_results/native_four_editions_20261011'
R = Path(__file__).resolve().parents[1] / 'evidence'
O = Path(__file__).resolve().parents[1]
BACKENDS = ('cpu_cpp', 'cpu_optimized', 'native_gpu', 'native_optimized')
NAMES = ('displace_y', 'displace_x', 'DPC_y', 'DPC_x', 'phase',
         'transmission', 'match_score_neg_ssd')
COMMON = NAMES[:5]

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def stamp(folder, stage, **extra):
    write(folder / 'stage.json', dict(stage=stage, pid=os.getpid(), time=time.time(), **extra))

def processes():
    out = {}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / 'stat').read_text().rsplit(')', 1)[1].split()
            out[int(d.name)] = dict(parent=int(stat[1]), ticks=int(stat[11])+int(stat[12]),
                                   cmd=(d/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
        except (OSError, ValueError, IndexError):
            pass
    return out

def owned(procs):
    ids = {os.getpid()}
    changed = True
    while changed:
        more = {pid for pid,v in procs.items() if v['parent'] in ids}
        changed = not more.issubset(ids)
        ids.update(more)
    return ids

def snapshot():
    procs = processes()
    ids = owned(procs)
    query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], text=True)
    gpu = [int(line.split(',')[0]) for line in query.splitlines() if line.strip()]
    mem = {line.split(':')[0]:int(line.split()[1])*1024
           for line in Path('/proc/meminfo').read_text().splitlines()}
    training = {pid:v['cmd'] for pid,v in procs.items() if pid not in ids and
                any(s in v['cmd'] for s in ('control_j35_speed_resume_20261012.py',
                    'j35_native_detail_retention_20261012_r2_speed_resume')) and
                'ssh ' not in v['cmd'] and 'bash -c' not in v['cmd']}
    return dict(time=time.time(), load=list(os.getloadavg()), available_bytes=mem['MemAvailable'],
                foreign_gpu=[pid for pid in gpu if pid not in ids], training=training), procs, ids

def wait_idle():
    first = None
    while True:
        state, procs, ids = snapshot()
        busy = {pid:v['cmd'] for pid,v in procs.items() if pid not in ids and
                any(s in v['cmd'].lower() for s in ('bench_full.py', 'run_native_bench.py',
                    'torchrun', 'train.py', 'train_j35', 'preprocess_dataset'))}
        ok = not state['foreign_gpu'] and not state['training'] and not busy and \
             state['load'][0] < 8 and state['available_bytes'] >= 40*1024**3
        state.update(idle=ok, external_data_tasks=busy)
        write(R/'load_latest.json', state)
        with (R/'load.jsonl').open('a') as f:
            f.write(json.dumps(state)+'\n')
        if ok:
            if first is None:
                first = time.monotonic()
            elif time.monotonic()-first >= 30:
                return
        else:
            first = None
            write(R/'status.json', dict(state='waiting_idle', evidence=state))
        time.sleep(5)

def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def worker(args):
    folder = R / args.tag
    folder.mkdir(parents=True, exist_ok=False)
    stamp(folder, 'load_inputs')
    arrays = [np.load(E/'inputs'/f'{args.mode}_{name}.npy', mmap_mode='r')
              for name in ('sample', 'reference')]
    h0,w0 = arrays[0].shape[-2:]
    y,x = (h0-args.size)//2, (w0-args.size)//2
    s,r = [np.ascontiguousarray(a[..., y:y+args.size, x:x+args.size], dtype=np.float32)
           for a in arrays]
    del arrays
    assert s.flags.c_contiguous and r.flags.c_contiguous
    input_hash_before = [hashlib.sha256(memoryview(a).cast('B')).hexdigest() for a in (s,r)]
    s.flags.writeable=False; r.flags.writeable=False
    shape = s.shape[-2:]
    backend = args.backend
    source = {}
    if backend in ('python', 'fastxst_c'):
        sys.path.insert(0, str(E/'release/python_reference/src' if backend=='python' else P/'src'))
        module = importlib.import_module(f'wsvt.{"reference" if backend=="python" else "fastxst"}.{args.mode.upper()}')
        cls = getattr(module, args.mode.upper())
        source['module'] = module.__file__
        if backend == 'fastxst_c':
            assert module.USE_C_SOLVER, 'FastXST C library did not load; no fallback permitted'
            source['c_library'] = module.c_lib._name
        else:
            dist = importlib.import_module('wsvt.reference.euclidean_dist').dist_numba
            # Original kernel has an explicit float64-only Numba signature.
            dist(np.zeros(8,np.float64), np.zeros((2,2,8),np.float64))
        # 128-square coarse image has 66 rows; 32*4 task groups creates
        # empty chunks in the unmodified original implementation. Smoke only.
        kw = dict(cal_half_window=10,N_s_extend=4,n_cores=8 if args.size==128 else 32,n_group=4,
                  energy=12400.,p_x=1.625e-6,z=.4845,wavelet_level_cut=1,
                  pyramid_level=1,n_iter=1,use_wavelet=True)
        kw['crop' if backend=='fastxst_c' and args.mode=='wsvt' else 'M_image'] = args.size
        if args.mode=='wxst':
            kw['N_s'] = 3
        if backend=='fastxst_c':
            kw.update(use_GPU=0,use_c=True,use_fft_corr=False)
            if args.mode=='wsvt':
                kw.update(N_template=0,Mag_factor=1,calc_T_D=False,calc_darkfield=False)
        def call():
            t=time.perf_counter(); q=cls(s,r,**kw); ctor=time.perf_counter()-t
            q.run(result_path=None)
            out=dict(zip(NAMES[:6], (q.displace[0],q.displace[1],q.DPC[0],q.DPC[1],q.phase,q.transmission)))
            score=getattr(q,'darkfield_nd',None)
            if score is not None:
                out[NAMES[6]]=score
            elapsed=time.perf_counter()-t
            info=dict(constructor_s=ctor, reported_solver_s=float(getattr(q,'time_cost',0.)), parameters=kw)
            return out,elapsed,info
    elif backend in ('cpu_cpp','cpu_optimized'):
        path=P/'bench_results/edge_gpu_workstation_20261011/cpu_build/libcpu_driver.so'
        if backend=='cpu_optimized': path=O/'build_cpu/libcpu_driver.so'
        lib=C.CDLL(str(path)); source['library']=str(path)
        lib.cpu_solve.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t,C.c_size_t,C.c_int,C.c_int]
        lib.cpu_solve.restype=C.c_void_p
        lib.cpu_error.restype=C.c_char_p
        lib.cpu_field.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
        lib.cpu_field.restype=C.POINTER(C.c_float)
        lib.cpu_times.argtypes=[C.c_void_p];lib.cpu_times.restype=C.POINTER(C.c_double)
        lib.cpu_free.argtypes=[C.c_void_p]
        def call():
            t=time.perf_counter()
            handle=lib.cpu_solve(int(args.mode=='wxst'),s.ctypes.data,r.ctypes.data,
                                 s.shape[0] if args.mode=='wsvt' else 1,*shape,32,4)
            if not handle:
                raise RuntimeError(lib.cpu_error().decode())
            try:
                out={}
                for i,n in enumerate(NAMES):
                    h,w=C.c_size_t(),C.c_size_t()
                    ptr=lib.cpu_field(handle,i,C.byref(h),C.byref(w))
                    out[n]=np.ctypeslib.as_array(ptr,shape=(h.value*w.value,)).reshape(h.value,w.value).copy()
                elapsed=time.perf_counter()-t
                times=list(np.ctypeslib.as_array(lib.cpu_times(handle),shape=(10,)))
                return out,elapsed,dict(copy_s=times[0],constructor_s=times[1],
                    solver_wall_s=times[2],reported_solver_s=times[3],stage_s=times[4:9],c_api_s=times[9])
            finally:
                lib.cpu_free(handle)
    else:
        sys.path.insert(0,str(E/'release/tools'))
        from native_api import Library,Session
        if backend in ('native_gpu','native_optimized'):
            path=E/'release/build_desktop/lib/libspeckle_native.so'
            if backend=='native_optimized': path=O/'build_gpu/lib/libspeckle_native.so'
            lib=Library(path);source['library']=str(path)
            make=lambda:Session(lib,shape,args.mode,'float32')
        else:
            import torch
            torch.set_num_threads(2)
            path=E/'baseline/speckle_edge/runtime.py';m=import_file('old_edge_runtime',path)
            source['module']=str(path)
            make=lambda:m.EdgeSession(shape,args.mode,'baseline' if args.mode=='wsvt' else 'fused',
                                     spline='gpu',input_dtype='float32',cpu_threads=2)
        persistent=make() if args.hot else None
        def call():
            t=time.perf_counter();q=persistent if args.hot else make();ctor=time.perf_counter()-t
            try:
                rt=time.perf_counter(); ref=q.prepare_reference(r); ref_wall=time.perf_counter()-rt
                st=time.perf_counter();out,info=q.solve(s); sample_wall=time.perf_counter()-st
                info.update(reference_api_s=ref_wall,sample_api_s=sample_wall,
                    host_validation_and_reference_dispatch_s=ref_wall-ref,
                    host_validation_and_sample_dispatch_s=sample_wall-info['sample_solver_s']-info['download_s'])
                if backend=='old_gpu':
                    torch.cuda.synchronize()
                elapsed=time.perf_counter()-t
                # Old wrapper uses darkfield_nd for negative SSD, not physical dark field.
                if 'darkfield_nd' in out:
                    out[NAMES[6]]=out.pop('darkfield_nd')
                info.update(constructor_s=ctor,reference_rebuild_s=ref)
                return out,elapsed,info
            finally:
                if not args.hot:
                    q.close()
    for label,path in list(source.items()):
        source[label+'_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    source.update(python_executable=sys.executable,numpy=np.__version__)
    write(folder/'source.json',source)
    stamp(folder,'warmup')
    out,elapsed,info=call();del out;gc.collect()
    warm=dict(array_api_s=elapsed,detail=info)
    rows=[]
    for j in range(args.repeats):
        stamp(folder,'formal',repeat=j)
        out,elapsed,info=call()
        assert set(COMMON).issubset(out), list(out)
        assert all(a.size and np.isfinite(a).all() for a in out.values())
        assert all(out[n].shape==(args.size-20,args.size-20) for n in COMMON)
        row=dict(repeat=j,array_api_s=elapsed,detail=info,
                 fields={n:dict(shape=list(a.shape),dtype=str(a.dtype)) for n,a in out.items()})
        rows.append(row);write(folder/'records.json',rows)
        with (folder/'timing.csv').open('w',newline='') as f:
            csvw=csv.writer(f);csvw.writerow(('repeat','array_api_s'))
            csvw.writerows((v['repeat'],v['array_api_s']) for v in rows)
        print('FORMAL',args.tag,j,elapsed,flush=True)
        if j==0:
            stamp(folder,'save_representative')
            for n,a in out.items():
                if args.size==2048 and not args.tag.startswith('probe') and not args.hot:
                    np.save(folder/(n+'.npy'),a,allow_pickle=False)
                elif args.hot:
                    np.save(folder/(n+'_roi.npy'),a[::16,::16],allow_pickle=False)
                else:
                    # Probe retains common outputs only; smoke never saves full-size duplicates.
                    if n in NAMES:
                        np.save(folder/(n+'.npy'),a,allow_pickle=False)
        del out;gc.collect()
    if backend.endswith('gpu') and args.hot:
        persistent.close()
    input_hash_after = [hashlib.sha256(memoryview(a).cast('B')).hexdigest() for a in (s,r)]
    assert input_hash_before == input_hash_after, 'Input mutation'
    process_memory = {line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmHWM:','VmRSS:'))}
    report=dict(input_sha256=input_hash_before, input_unchanged=True, process_memory=process_memory, backend=backend,mode=args.mode,size=args.size,hot=args.hot,
        input_shape=list(s.shape),input_dtype='float32',input_transport='pageable',source=source,
        warmup=warm,records=rows,mean_s=statistics.mean(v['array_api_s'] for v in rows),
        sample_sd_s=statistics.stdev(v['array_api_s'] for v in rows) if len(rows)>1 else None,
        boundary='Ready host arrays to independently owned host results; fresh object/session including init and reference rebuild; no file I/O. Destruction after clock. Imports and JIT warmup excluded.' if not args.hot else 'Persistent session, reference rebuilt each pair; ready pageable float32 host arrays to owned host results.')
    write(folder/'report.json',report);stamp(folder,'complete')

def compare(a,b,path):
    report={}
    for n in COMMON:
        aa=np.load(a/(n+'.npy')).astype(np.float64);bb=np.load(b/(n+'.npy')).astype(np.float64)
        d=aa-bb;mx=float(np.max(np.abs(d)));rm=float(np.sqrt(np.mean(d*d)))
        scale=float(np.max(np.abs(aa)));rms=float(np.sqrt(np.mean(aa*aa)))
        gate=(mx<=2e-3+1e-5*scale and rm<=2e-4+1e-6*rms) if n=='phase' else mx==0
        report[n]=dict(max_abs=mx,rmse=rm,pass_gate=gate,
                       central_row_a=aa[aa.shape[0]//2,::max(1,aa.shape[1]//128)].tolist(),
                       central_row_b=bb[bb.shape[0]//2,::max(1,bb.shape[1]//128)].tolist())
    write(path,dict(reference=str(a),candidate=str(b),fields=report,
        pass_gate=all(v['pass_gate'] for v in report.values()),
        scope='Engineering old/native GPU agreement, not independent physical accuracy.'))
    return all(v['pass_gate'] for v in report.values())

def unit(tag,backend,mode,size,repeats,hot=False):
    folder=R/tag
    if (folder/'invalid_competition.json').exists():
        raise RuntimeError('Invalidated unit is retained and cannot enter summary: '+tag)
    if (folder/'report.json').exists():
        return json.loads((folder/'report.json').read_text())
    if folder.exists():
        raise RuntimeError('Incomplete unit retained; manual diagnosis required: '+tag)
    wait_idle()
    write(R/'status.json',dict(state='running',unit=tag,time=time.time()))
    py='/home/yrc/vscodespace/anaconda3/'+('envs/speckle/bin/python' if backend in ('python','fastxst_c') else 'bin/python')
    cmd=[py,'-u',str(Path(__file__)),'worker','--tag',tag,'--backend',backend,
         '--mode',mode,'--size',str(size),'--repeats',str(repeats)]
    if hot:cmd.append('--hot')
    write(R/(tag+'_command.json'),cmd)
    samples=[];competition=[];prev=processes();before=time.monotonic()
    with (R/(tag+'.log')).open('w') as log:
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='32',OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
        if backend in ('cpu_cpp','cpu_optimized'):
            env['LD_LIBRARY_PATH']='/home/yrc/vscodespace/anaconda3/envs/wsvt_cpp/lib:'+env.get('LD_LIBRARY_PATH','')
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(P),env=env)
        while p.poll() is None:
            time.sleep(2)
            state,now,ids=snapshot();dt=time.monotonic()-before
            heavy={pid:v['cmd'] for pid,v in now.items() if pid not in ids and pid in prev and
                   (v['ticks']-prev[pid]['ticks'])/os.sysconf('SC_CLK_TCK')/dt>1.0}
            state.update(external_cpu_over_one_core=heavy)
            samples.append(state)
            if state['foreign_gpu'] or state['training'] or heavy:
                competition.append(state)
            prev=now;before=time.monotonic()
    write(R/(tag+'_load.json'),samples)
    if competition:
        write(folder/'invalid_competition.json',competition)
        raise RuntimeError('Competing task detected; unit excluded. Other tasks untouched: '+tag)
    if p.returncode:
        raise RuntimeError('Worker failed; retained log: '+tag)
    return json.loads((folder/'report.json').read_text())

def compare_all(mode,kind,size):
    old=R/f'{kind}_{mode}_{"cpu_cpp" if kind.endswith("cpu") else "native_gpu"}'
    new=R/f'{kind}_{mode}_{"cpu_optimized" if kind.endswith("cpu") else "native_optimized"}'
    fields={}
    for n in NAMES:
        aa=np.load(old/(n+'.npy'));bb=np.load(new/(n+'.npy'))
        assert aa.shape==bb.shape
        d=aa.astype(np.float64)-bb.astype(np.float64)
        fields[n]=dict(bitwise_equal=bool(np.array_equal(aa.view(np.uint32),bb.view(np.uint32))),
            max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))))
    ok=all(v['bitwise_equal'] for v in fields.values())
    write(R/f'{kind}_{mode}_check.json',dict(pass_gate=ok,fields=fields,
        input_size=size,scope='Unchanged numerical reconstruction rules; not a physical-truth test.'))
    assert ok, 'Original/optimized numerical change'

def controller():
    import fcntl
    lock=(R/'single_instance.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for mode in ('wsvt','wxst'):
        for base,new,kind in [('cpu_cpp','cpu_optimized','smoke_cpu'),('native_gpu','native_optimized','smoke_gpu')]:
            unit(f'{kind}_{mode}_{base}',base,mode,128,1)
            unit(f'{kind}_{mode}_{new}',new,mode,128,1)
            compare_all(mode,kind,128)
    reports=[]
    for mode in ('wsvt','wxst'):
        for base,new,kind in [('cpu_cpp','cpu_optimized','formal_cpu'),('native_gpu','native_optimized','formal_gpu')]:
            for backend in (base,new):
                report=unit(f'{kind}_{mode}_{backend}',backend,mode,2048,5)
                reports.append(report);write(R/'summary.json',reports)
                with (R/'summary.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(('mode','backend','size','n','array_api_mean_s','sample_sd_s','output_fields'))
                    for rr in reports:
                        w.writerow((rr['mode'],rr['backend'],rr['size'],len(rr['records']),rr['mean_s'],rr['sample_sd_s'],len(rr['records'][0]['fields'])))
            compare_all(mode,kind,2048)
    write(R/'status.json',dict(state='complete',time=time.time(),formal_units=8,formal_trials=40))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=('controller','worker'))
    ap.add_argument('--backend',choices=BACKENDS);ap.add_argument('--mode',choices=('wsvt','wxst'))
    ap.add_argument('--size',type=int);ap.add_argument('--repeats',type=int);ap.add_argument('--tag');ap.add_argument('--hot',action='store_true')
    args=ap.parse_args()
    try:
        controller() if args.task=='controller' else worker(args)
    except BlockingIOError:
        print('An existing controller owns the lock; no duplicate started.',flush=True)
        sys.exit(2)
    except Exception as exc:
        traceback.print_exc()
        if args.task=='controller':
            write(R/'status.json',dict(state='failed',error=str(exc),time=time.time()))
        sys.exit(1)
