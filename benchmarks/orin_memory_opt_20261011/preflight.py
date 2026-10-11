"""Read-only source/data/device audit; preserves original package and experiment inputs."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent;ORIG=R.parent/'edge_speckle';DATA=R.parent/'edge_adaptation_20261011/wsvt100_full2048.npz'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def command(a):
    p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
manifest=json.loads((ORIG/'docs/release_manifest.json').read_text())['files_sha256']
checked={n:dict(expected=h,actual=sha(ORIG/n)) for n,h in manifest.items()}
assert all(r['expected']==r['actual'] for r in checked.values()),'Original source differs from frozen manifest'
native={p.relative_to(ORIG).as_posix():sha(p) for p in (ORIG/'speckle_edge/_native').glob('*') if p.is_file()}
report=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=command(['hostname']),cpu=command(['lscpu']),
    before_load=os.getloadavg(),meminfo=Path('/proc/meminfo').read_text(),disk=command(['df','-h',str(R)]),
    running_processes=command(['ps','-eo','pid,pcpu,comm']),power_mode=command(['/usr/sbin/nvpmodel','-q']),
    tegrastats_idle=command(['timeout','3','tegrastats','--interval','1000']),
    original_path=str(ORIG),original_source_checks=checked,original_native_sha256=native,
    original_build_flags=(ORIG/'build/CMakeFiles/edge_kernels.dir/flags.make').read_text(),
    original_build_info=json.loads((ORIG/'speckle_edge/_native/build_info.json').read_text()),
    data_npz_path=str(DATA),data_npz_sha256=sha(DATA),data_npz_bytes=DATA.stat().st_size)
(R/'preflight.json').write_text(json.dumps(report,indent=2))
import numpy as np
with np.load(DATA,allow_pickle=False) as z:
    arrays={n:z[n] for n in ('sample','reference')}
    values={}
    for n,a in arrays.items():
        assert a.shape==(100,2048,2048) and a.dtype==np.uint16 and a.flags.c_contiguous
        values[n]=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest(),
            per_frame_sha256=[hashlib.sha256(memoryview(a[j]).cast('B')).hexdigest() for j in range(100)])
    report['arrays']=values
(R/'preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(original_source_manifest_pass=True,arrays={k:{n:v[n] for n in ('shape','dtype','sha256')} for k,v in values.items()},original_native_sha256=native),indent=2),flush=True)
