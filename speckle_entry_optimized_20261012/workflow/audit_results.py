from pathlib import Path
import csv,json,statistics as st
import numpy as np
root=Path(__file__).resolve().parents[1];e=root/'evidence'
rows=[];raw=[];checks={};rois={}
for mode in ('wsvt','wxst'):
    for kind,base,new in [('cpu','cpu_cpp','cpu_optimized'),('gpu','native_gpu','native_optimized')]:
        check=json.loads((e/f'formal_{kind}_{mode}_check.json').read_text())
        assert check['pass_gate'];checks[f'{mode}_{kind}']=check
        for backend in (base,new):
            folder=e/f'formal_{kind}_{mode}_{backend}'
            r=json.loads((folder/'report.json').read_text())
            assert r['input_unchanged'] and len(r['records'])==5
            fields=[v['fields'] for v in r['records']];assert all(len(f)==7 for f in fields)
            detail={k:st.mean(v['detail'][k] for v in r['records']) for k,val in r['records'][0]['detail'].items() if isinstance(val,(float,int))}
            row=dict(mode=mode,backend=backend,array_api_mean_s=r['mean_s'],array_api_sd_s=r['sample_sd_s'],
                n=5,detail_means=detail,process_memory=r['process_memory'],input_sha256=r['input_sha256'])
            rows.append(row)
            for v in r['records']:
                raw.append(dict(mode=mode,backend=backend,repeat=v['repeat'],array_api_s=v['array_api_s'],**v['detail']))
            for field in ('phase','displace_x','displace_y','DPC_x','DPC_y'):
                a=np.load(folder/(field+'.npy'),mmap_mode='r');h,w=a.shape;y,x=(h-512)//2,(w-512)//2
                rois[f'{mode}_{backend}_{field}']=a[y:y+512,x:x+512].copy()
            for field in ('phase','displace_x','displace_y'):
                a=np.load(folder/(field+'.npy'),mmap_mode='r')
                rois[f'{mode}_{backend}_{field}_profile']=a[a.shape[0]//2,:].copy()
for path in e.glob('*_load.json'):
    for v in json.loads(path.read_text()):
        assert not v['foreign_gpu'] and not v['training'] and not v.get('external_cpu_over_one_core')
for mode in ('wsvt','wxst'):
    hashes={tuple(r['input_sha256']) for r in rows if r['mode']==mode}
    assert len(hashes)==1,'Backends used different input content'
np.savez_compressed(e/'comparison_rois.npz',**rois)
(e/'audit.json').write_text(json.dumps(dict(rows=rows,checks=checks,
    all_gate_pass=True,formal_trials=40,competition_detected=False,
    phase_roi=dict(output_y=[758,1270],output_x=[758,1270],raw_origin_offset=10),
    boundary='Cold complete array API, float32 pageable, reference rebuilt, seven independent host results; no file I/O/destruction.',
    notes=['CPU copy_s ABI slot is O(1) view setup in optimized entry, not an array copy.',
           'Reported CPU solver and solver() wall clocks differ; both are retained separately.',
           'GPU sample_solver_s excludes reference, init, finite validation and output download.',
           'VmHWM is sampled whole-process high-water RSS including ready input/output arrays, not isolated algorithm allocation.']),indent=2))
keys=['mode','backend','repeat','array_api_s']+sorted({k for r in raw for k in r if k not in ('mode','backend','repeat','array_api_s') and k!='stage_s' and k!='stages_s'})
with (e/'raw_timing.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(raw)
print(json.dumps([dict(mode=r['mode'],backend=r['backend'],mean=r['array_api_mean_s'],sd=r['array_api_sd_s'],detail=r['detail_means']) for r in rows],indent=2))
