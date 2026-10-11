from pathlib import Path
import hashlib,json,shutil,subprocess,sys,zipfile
R=Path(__file__).resolve().parent
pre=json.loads((R/'preflight.json').read_text());expected=json.loads((R/'server_array_hashes.json').read_text())
for n in ('sample','reference'):
    assert pre['arrays'][n]['sha256']==expected[n]['sha256'],n
assert hashlib.sha256((R/'source.zip').read_bytes()).hexdigest()=='8577c1006d9b9579e66c8c77c2bd476c8da21cdbe44a868377cfa58f9c93b1c9'
shutil.copytree(R.parent/'edge_speckle',R/'baseline',ignore=shutil.ignore_patterns('build','dist','__pycache__'))
with zipfile.ZipFile(R/'source.zip') as z:
    names=z.namelist();print(names[:5],flush=True)
    for name in names:assert not name.startswith('/') and '..' not in Path(name).parts
    z.extractall(R/'extracted')
roots=list((R/'extracted').rglob('pyproject.toml'));assert len(roots)==1,roots
shutil.move(str(roots[0].parent),str(R/'package'))
subprocess.run([sys.executable,'tools/check_sources.py','--baseline',str(R/'baseline')],cwd=R/'package',check=True)
subprocess.run([sys.executable,'tools/build.py','--arch','87','--nvcc','/usr/local/cuda/bin/nvcc','--jobs','2'],cwd=R/'package',check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=R/'package',check=True)
print('DEPLOY_COMPLETE',flush=True)
