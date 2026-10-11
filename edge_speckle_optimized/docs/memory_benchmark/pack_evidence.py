"""Lightweight evidence only; full arrays stay on the server."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
R=Path(__file__).resolve().parent;P=R/'package';B=R/'gpu_build';OLD=R.parent/'edge_gpu_workstation_20261011/package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(a):return subprocess.check_output(a,text=True).strip()
cache=(B/'CMakeCache.txt').read_text();flags=(B/'CMakeFiles/edge_kernels.dir/flags.make').read_text()
manifest=json.loads((P/'docs/release_manifest.json').read_text())['files_sha256']
assert all(sha(P/n)==h for n,h in manifest.items())
assert '--fmad=false' in flags and 'sm_89' in flags
old=(OLD/'native/edge_kernels.cu').read_text();new=(P/'native/edge_kernels.cu').read_text();marker='template<class T> __global__ void stats100';assert old.split(marker)[0]==new.split(marker)[0]
receipt=dict(native_cuda=cmd(['/usr/local/cuda-12.8/bin/nvcc','--version']),gcc=cmd(['/usr/bin/g++-12','--version']),cmake=cmd(['/home/yrc/vscodespace/anaconda3/bin/cmake','--version']),
    cmake_cache=cache,cuda_flags=flags,source_archive_sha256=sha(R/'final_source_0.1.1.zip'),candidate_source_hashes=manifest,
    native_binary_sha256={p.name:sha(p) for p in (P/'speckle_edge/_native').glob('*.so')},baseline_binary_sha256=sha(OLD/'speckle_edge/_native/libedge_kernels.so'),
    matching_subpixel_spline_block_sha256=hashlib.sha256(new.split(marker)[0].encode()).hexdigest(),matching_subpixel_spline_unchanged=True,
    build_scope='independent directory; CUDA12.8.93 GCC12.3 SM89, --fmad=false. Native sources unchanged since measured binary build; final archive adds CLI/default16/docs/configs.')
(R/'build_receipt.json').write_text(json.dumps(receipt,indent=2));(P/'speckle_edge/_native/build_info.json').write_text(json.dumps(receipt,indent=2))
names=('smoke.json','smoke.log','final_smoke.json','final_smoke.log','wsvt.log','wxst.log','controlled_wsvt.log','controlled_wxst.log','run_experiment.py','run_controlled.py','refine_measurement.py','final_smoke.py','plot_results.py','autoresearch.md','build_receipt.json')
files=[R/n for n in names]
for folder in ('full_wsvt','full_wxst','controlled_wsvt','controlled_wxst'):
    files.extend(p for p in (R/folder).iterdir() if p.suffix in ('.json','.csv'))
files.extend((R/'figures').iterdir())
with zipfile.ZipFile(R/'lightweight_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(R))
print(json.dumps(dict(archive=str(R/'lightweight_results.zip'),sha256=sha(R/'lightweight_results.zip'),files=len(files),bytes=(R/'lightweight_results.zip').stat().st_size)))
