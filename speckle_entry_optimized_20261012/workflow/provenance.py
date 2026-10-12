from pathlib import Path
import hashlib,json,subprocess,time
root=Path(__file__).resolve().parents[1]
base=root.parent/'native_four_editions_20261011/release'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
files={};changes={}
for sub in ('cpu_cpp/source','gpu_common','tools'):
    for p in sorted((root/sub).rglob('*')):
        if not p.is_file() or '__pycache__' in str(p):continue
        rel=str(p.relative_to(root));old=base/rel
        files[rel]=sha(p)
        if not old.exists() or files[rel]!=sha(old):
            changes[rel]=dict(original_sha256=sha(old) if old.exists() else None,optimized_sha256=files[rel])
paths=[root/'build_cpu/libcpu_driver.so',root/'build_gpu/lib/libspeckle_native.so',
       base/'build_desktop/lib/libspeckle_native.so',
       root.parent/'edge_gpu_workstation_20261011/cpu_build/libcpu_driver.so']
receipt=dict(time=time.time(),source_sha256=files,changed_files=changes,
    binaries={str(p):sha(p) for p in paths},
    caches={n:(root/n/'CMakeCache.txt').read_text() for n in ('build_cpu','build_gpu')},
    cpu_driver_original_sha256=sha(root/'cpu_cpp/cpu_driver_original.cpp'),
    cpu_driver_optimized_sha256=sha(root/'cpu_cpp/cpu_driver.cpp'),
    hardware=subprocess.check_output(['bash','-c','hostname; lscpu; nvidia-smi'],text=True))
e=root/'evidence';e.mkdir(exist_ok=True)
(e/'build_receipt.json').write_text(json.dumps(receipt,indent=2))
print('Changed source files:',*changes,sep='\n')
