from pathlib import Path
root=Path(__file__).resolve().parents[1]
old=Path('C:/Users/Yangrc/Documents/Codex/2026-10-07/ppt/workflow/matched_speed.py')
s=old.read_text(encoding='utf-8')
s=s.replace("R = Path(__file__).resolve().parent", "R = Path(__file__).resolve().parents[1] / 'evidence'\nO = Path(__file__).resolve().parents[1]")
s=s.replace("BACKENDS = ('python', 'fastxst_c', 'cpu_cpp', 'old_gpu', 'native_gpu')", "BACKENDS = ('cpu_cpp', 'cpu_optimized', 'native_gpu', 'native_optimized')")
s=s.replace("    shape = s.shape[-2:]", "    input_hash_before = [hashlib.sha256(memoryview(a).cast('B')).hexdigest() for a in (s,r)]\n    s.flags.writeable=False; r.flags.writeable=False\n    shape = s.shape[-2:]")
s=s.replace("elif backend=='cpu_cpp':", "elif backend in ('cpu_cpp','cpu_optimized'):")
s=s.replace("        lib=C.CDLL(str(path));", "        if backend=='cpu_optimized': path=O/'build_cpu/libcpu_driver.so'\n        lib=C.CDLL(str(path));",1)
s=s.replace("if backend=='native_gpu':", "if backend in ('native_gpu','native_optimized'):")
s=s.replace("            lib=Library(path);", "            if backend=='native_optimized': path=O/'build_gpu/lib/libspeckle_native.so'\n            lib=Library(path);",1)
s=s.replace("                ref=q.prepare_reference(r);out,info=q.solve(s)", "                rt=time.perf_counter(); ref=q.prepare_reference(r); ref_wall=time.perf_counter()-rt\n                st=time.perf_counter();out,info=q.solve(s); sample_wall=time.perf_counter()-st\n                info.update(reference_api_s=ref_wall,sample_api_s=sample_wall,\n                    host_validation_and_reference_dispatch_s=ref_wall-ref,\n                    host_validation_and_sample_dispatch_s=sample_wall-info['sample_solver_s']-info['download_s'])")
s=s.replace("                    if n in COMMON:", "                    if n in NAMES:")
s=s.replace("    report=dict(backend=backend", "    input_hash_after = [hashlib.sha256(memoryview(a).cast('B')).hexdigest() for a in (s,r)]\n    assert input_hash_before == input_hash_after, 'Input mutation'\n    process_memory = {line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmHWM:','VmRSS:'))}\n    report=dict(input_sha256=input_hash_before, input_unchanged=True, process_memory=process_memory, backend=backend")
s=s.replace("if backend=='cpu_cpp':\n            env", "if backend in ('cpu_cpp','cpu_optimized'):\n            env")
begin=s.index('def controller():'); end=s.index("if __name__=='__main__':")
s=s[:begin]+'''def compare_all(mode,kind,size):
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

''' + s[end:]
(root/'workflow/matched_entry.py').write_text(s,encoding='utf-8')
