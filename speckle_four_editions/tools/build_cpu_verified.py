"""Build copied frozen CPU sources; isolate dependencies and retain compiler receipt."""
from pathlib import Path
import hashlib,json,os,platform,subprocess,shutil
R=Path(__file__).resolve().parents[1];orin=platform.machine()=='aarch64';b=R/('build_cpu_jetson' if orin else 'build_cpu_desktop');env=os.environ.copy()
if orin:
    prefix=Path('/home/xray/Documents/WSVT_GPU_orin_20261010/cpu_compare_20261011/toolchain');cmake=str(prefix/'bin/cmake');cxx=str(prefix/'bin/aarch64-conda-linux-gnu-g++');cc=str(prefix/'bin/aarch64-conda-linux-gnu-gcc');h5='OFF'
else:
    prefix=Path('/home/yrc/vscodespace/anaconda3/envs/wsvt_cpp');cmake=str(prefix/'bin/cmake');cxx='/usr/bin/c++';cc='/usr/bin/cc';h5='ON'
env['PATH']=str(prefix/'bin')+os.pathsep+env['PATH'];env['CONDA_PREFIX']=str(prefix);env['LD_LIBRARY_PATH']=str(prefix/'lib');env['PKG_CONFIG_PATH']=str(prefix/'lib/pkgconfig')
if not Path(cmake).exists():cmake=shutil.which('cmake',path=env['PATH'])
cmd=[cmake,'-S',str(R/'cpu_cpp/source'),'-B',str(b),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+cxx,'-DCMAKE_C_COMPILER='+cc,'-DWSVT_WITH_OPENCV=OFF','-DWSVT_WITH_HDF5='+h5,'-DFFTW3F_LIB='+str(prefix/'lib/libfftw3f.so'),'-DFFTW3F_INC='+str(prefix/'include'),'-DFFTW3F_THREADS_LIB='+str(prefix/'lib/libfftw3f_threads.so'),'-DCMAKE_PREFIX_PATH='+str(prefix)]
p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(p.stdout,flush=True);p.check_returncode();assert 'FFTW3 (single precision) enabled' in p.stdout
subprocess.run([cmake,'--build',str(b),'--target','wsvt_cli','-j','2'],env=env,check=True)
if orin:env['LD_PRELOAD']=str(prefix/'lib/libstdc++.so.6')
help_result=subprocess.run([str(b/'wsvt_cli'),'--help'],env=env,text=True,capture_output=True);assert help_result.returncode==2 and 'usage:' in help_result.stdout
demo=subprocess.run([str(b/'wsvt_cli'),'demo'],env=env,text=True,capture_output=True);print(demo.stdout,flush=True);demo.check_returncode()
receipt=dict(platform=platform.machine(),configure=cmd,compiler_version=subprocess.check_output([cxx,'--version'],env=env,text=True),binary_sha256=hashlib.sha256((b/'wsvt_cli').read_bytes()).hexdigest(),ldd=subprocess.check_output(['ldd',str(b/'wsvt_cli')],env=env,text=True),demo_stdout=demo.stdout,fft='FFTW3 float with thread library, no brute DFT fallback',snapshot='Frozen53 original files unchanged; missing packaging template copied from baseline artifact')
(R/'docs'/('build_cpu_jetson.json' if orin else 'build_cpu_desktop.json')).write_text(json.dumps(receipt,indent=2));print('CPU_EDITION_BUILD_PASS',flush=True)
