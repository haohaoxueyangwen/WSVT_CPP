"""Audit baseline ancestry, unchanged algorithm blocks and final source manifest."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,default=ROOT.parent/'edge_speckle');a=ap.parse_args()
    origins=json.loads((ROOT/'docs/optimization_origins.json').read_text());rows=[]
    for n,v in origins['changed'].items():
        rows.append(dict(file=n,baseline_preserved=sha(a.baseline/n)==v['baseline_sha256'],candidate_registered=sha(ROOT/n)==v['candidate_sha256']))
    for n,v in origins['unchanged'].items():
        if n=='docs/release_manifest.json':continue # Generated manifest, registered independently below.
        rows.append(dict(file=n,baseline_preserved=sha(a.baseline/n)==v,candidate_registered=sha(ROOT/n)==v))
    old=(a.baseline/'native/edge_kernels.cu').read_text();new=(ROOT/'native/edge_kernels.cu').read_text()
    # This entire block contains SSD matching, subpixel and spline kernels, unchanged.
    stop='template<class T> __global__ void stats100'
    block_equal=old.split(stop)[0]==new.split(stop)[0]
    post_start='__global__ void crop_sums';post_end='extern "C" int edge_match'
    post_equal=old.split(post_start)[1].split(post_end)[0]==new.split(post_start)[1].split(post_end)[0]
    old_runtime=(a.baseline/'speckle_edge/runtime.py').read_text()
    new_runtime=(ROOT/'speckle_edge/runtime.py').read_text()
    phase_equal=old_runtime.split('    def _phase(self):')[1].split('    def solve(')[0]==new_runtime.split('    def _phase(self):')[1].split('    def solve(')[0]
    manifest=json.loads((ROOT/'docs/release_manifest.json').read_text())['files_sha256']
    manifest_ok=all(sha(ROOT/n)==v for n,v in manifest.items())
    result=dict(pass_check=all(r['baseline_preserved'] and r['candidate_registered'] for r in rows) and block_equal and post_equal and phase_equal and manifest_ok,
                matching_subpixel_spline_block_unchanged=block_equal,postprocess_block_unchanged=post_equal,phase_block_unchanged=phase_equal,manifest_ok=manifest_ok,files=rows)
    print(json.dumps(result,indent=2))
    if not result['pass_check']:raise SystemExit(1)
if __name__=='__main__':main()
