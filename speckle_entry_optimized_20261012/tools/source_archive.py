from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parents[1];D=R/'dist';D.mkdir(exist_ok=True)
files=[p for p in R.rglob('*') if p.is_file() and not any(s.startswith('build_') or s in ('dist','__pycache__','runs') or s.startswith(('formal_','smoke_')) for s in p.relative_to(R).parts[:-1]) and p.suffix not in ('.so','.pyc','.npy','.npz','.gz','.zip')]
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='release_manifest.json'}
(R/'docs/release_manifest.json').write_text(json.dumps(manifest,indent=2));files=[p for p in files if p.name!='release_manifest.json']+[R/'docs/release_manifest.json']
path=D/'speckle-entry-optimized-20261012-source.zip'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
    for p in files:z.write(p,p.relative_to(R).as_posix())
print(path,hashlib.sha256(path.read_bytes()).hexdigest(),len(files),flush=True)
