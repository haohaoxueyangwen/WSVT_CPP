"""One-time TIFF-stack/NPZ to native NPY conversion; not a runtime dependency."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['wsvt','wxst']);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--npz',type=Path);ap.add_argument('--sample-key',default='sample');ap.add_argument('--reference-key',default='reference');ap.add_argument('--sample-glob');ap.add_argument('--reference-glob');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
if a.npz:
    with np.load(a.npz) as f:s=np.asarray(f[a.sample_key]);r=np.asarray(f[a.reference_key])
else:
    import glob,tifffile
    def read(pattern):
        files=sorted(glob.glob(pattern));assert files,'No files matching '+pattern
        if a.mode=='wxst':assert len(files)==1,'WXST requires one explicit file per side'
        values=[tifffile.imread(p) for p in files];return np.stack(values) if a.mode=='wsvt' else values[0]
    s=read(a.sample_glob);r=read(a.reference_glob)
assert s.shape==r.shape and s.ndim==(3 if a.mode=='wsvt' else 2)
if a.mode=='wsvt':assert s.shape[0]==100 and s.dtype==r.dtype==np.uint16,'Validated native WSVT preset expects100 uint16 frames'
else:s=s.astype(np.float32);r=r.astype(np.float32)
receipt={}
for name,v in (('sample',s),('reference',r)):
    v=np.ascontiguousarray(v);np.save(a.out/(name+'.npy'),v,allow_pickle=False);receipt[name]=dict(shape=list(v.shape),dtype=str(v.dtype),canonical_sha256=hashlib.sha256(memoryview(v).cast('B')).hexdigest())
(a.out/'input_manifest.json').write_text(json.dumps(receipt,indent=2))
