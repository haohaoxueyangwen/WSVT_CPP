"""Preserved Python solver behind an explicit .npy input/output command line."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['wsvt','wxst'])
    for name in ('sample','reference','out'):ap.add_argument('--'+name,required=True,type=Path)
    for name in ('pixel-m','distance-m','energy-ev'):ap.add_argument('--'+name,required=True,type=float)
    ap.add_argument('--cores',type=int,default=4);ap.add_argument('--groups',type=int,default=4);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);s=np.load(args.sample);r=np.load(args.reference)
    assert s.shape==r.shape and s.ndim==(3 if args.mode=='wsvt' else 2)
    kwargs=dict(M_image=s.shape[-1],cal_half_window=10,N_s_extend=4,n_cores=args.cores,n_group=args.groups,energy=args.energy_ev,p_x=args.pixel_m,z=args.distance_m,wavelet_level_cut=1,pyramid_level=2,n_iter=1,use_wavelet=True)
    if args.mode=='wsvt':
        from wsvt.reference.WSVT import WSVT
        q=WSVT(s,r,**kwargs)
    else:
        from wsvt.reference.WXST import WXST
        q=WXST(s,r,N_s=3,**kwargs)
    t=time.perf_counter();q.run();elapsed=time.perf_counter()-t
    out=dict(displace_y=q.displace[0],displace_x=q.displace[1],DPC_y=q.DPC[0],DPC_x=q.DPC[1],phase=q.phase)
    if hasattr(q,'transmission'):out['transmission']=q.transmission
    for name,a in out.items():np.save(args.out/(name+'.npy'),a,allow_pickle=False)
    (args.out/'report.json').write_text(json.dumps(dict(mode=args.mode,elapsed_run_s=elapsed,scope='Original q.run(), no file read/save; original output/normalization rules retained',fields=list(out),settings=kwargs),indent=2))
if __name__=='__main__':main()
