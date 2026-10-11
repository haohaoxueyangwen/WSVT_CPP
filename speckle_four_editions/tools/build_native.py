"""Build-time convenience only; installed executables do not require Python."""
from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--platform',choices=['desktop','jetson'],required=True);ap.add_argument('--nvcc',required=True);ap.add_argument('--jobs',type=int,default=2);a=ap.parse_args()
arch='89' if a.platform=='desktop' else '87';build=R/('build_'+a.platform);edition=R/('gpu_discrete' if a.platform=='desktop' else 'gpu_jetson')
command=['cmake','-S',str(edition),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER='+a.nvcc,'-DCUDA_TOOLKIT_ROOT_DIR='+str(Path(a.nvcc).parents[1]),'-DSPECKLE_ARCH='+arch]
subprocess.run(command,check=True);subprocess.run(['cmake','--build',str(build),'-j',str(a.jobs)],check=True)
files=[p for folder in ('gpu_common','gpu_discrete','gpu_jetson') for p in (R/folder).rglob('*') if p.is_file()]+list((build/'lib').glob('*.so'))+list((build/'bin').glob('*'))
report=dict(platform=a.platform,arch=arch,configure=command,nvcc=subprocess.check_output([a.nvcc,'--version'],text=True),cxx=subprocess.check_output(['c++','--version'],text=True),
    compile_flags=(build/'shared_gpu/CMakeFiles/speckle_native.dir/flags.make').read_text(),hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    runtime_dependencies=subprocess.check_output(['ldd',str(build/'bin/speckle_gpu')],text=True))
assert '--fmad=false' in report['compile_flags'] and '-arch=sm_'+arch in report['compile_flags']
assert 'libtorch' not in report['runtime_dependencies'] and 'libpython' not in report['runtime_dependencies']
(R/'docs'/('build_'+a.platform+'.json')).write_text(json.dumps(report,indent=2));print('NATIVE_BUILD_COMPLETE',a.platform,flush=True)
