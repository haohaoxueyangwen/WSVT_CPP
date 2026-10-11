"""Run beside remote results; complete fields, fixed physical ROIs and image checks."""
from pathlib import Path
import csv,hashlib,json,tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
NAMES=('displace_y','displace_x','DPC_y','DPC_x','phase','transmission','match_score_neg_ssd')
O=R/'analysis';O.mkdir(exist_ok=False)
comparisons={};rows=[]
for mode in ('wsvt','wxst'):
    summary=json.loads((R/(mode+'_summary.json')).read_text());comparisons[mode]={}
    for profile,v in summary.items():
        if 'summary' in v:
            s=v['summary']['solver_s'];a=v['summary']['api_s'];b=v['memory'].get('native_explicit_device_bytes',v['memory'].get('torch_peak_allocated_bytes'))
        else:s={'mean':v['solver_mean_s'],'sample_sd':v['solver_sample_sd_s']};a={'mean':v['api_mean_s'],'sample_sd':v['api_sample_sd_s']};b=v['explicit_device_bytes']
        m=v['sampled_memory'];rows.append(dict(mode=mode,profile=profile,solver_mean_s=s['mean'],solver_sample_sd_s=s['sample_sd'],api_mean_s=a['mean'],api_sample_sd_s=a['sample_sd'],explicit_native_or_torch_peak_GiB=b/2**30,nvml_process_peak_GiB=m['nvml_process_peak_bytes']/2**30 if m['nvml_process_peak_bytes'] else None,sampled_rss_GiB=m['peak_rss_bytes']/2**30,system_used_peak_GiB=m['system_used_peak_bytes']/2**30))
        if profile=='original':continue
        comp={}
        for name in NAMES:
            old=np.load(R/mode/'original'/(name+'.npy'),mmap_mode='r');new=np.load(R/mode/profile/(name+'.npy'),mmap_mode='r')
            assert old.shape==new.shape and np.isfinite(new).all()
            d=new.astype(np.float64)-old
            # Non-transmission valid array origin (10,10) in full input image.
            origin=0 if name=='transmission' else 10
            regions={'full':(slice(None),slice(None)), 'center1000':(slice(524-origin,1524-origin),slice(524-origin,1524-origin)), 'user_roi':(slice(924-origin,1424-origin),slice(724-origin,1324-origin))}
            comp[name]={}
            for region,sl in regions.items():
                delta=d[sl];base=old[sl].astype(np.float64);mx=float(np.max(np.abs(delta)));rms=float(np.sqrt(np.mean(delta*delta)));exact=bool(np.array_equal(old[sl],new[sl]))
                passed=exact if name!='phase' else mx<=.002+1e-5*np.max(np.abs(base)) and rms<=.0002+1e-6*np.sqrt(np.mean(base*base))
                comp[name][region]=dict(shape=list(delta.shape),exact=exact,max_abs=mx,rmse=rms,changed_pixels=int(np.count_nonzero(delta)),pass_gate=bool(passed))
            del old,new,d
        vfeatures=v.get('features_sha256',{});basefeatures=summary['original'].get('features_sha256',{})
        comp['descriptors']={k:val==basefeatures[k] for k,val in vfeatures.items() if k in basefeatures}
        comparisons[mode][profile]=comp
    selected=['original','cli_'+mode+'_speed_'+('mapped' if summary['original']['platform']=='jetson' and mode=='wsvt' else 'copy')]
    if mode=='wsvt':selected+=['cli_wsvt_low_memory_'+('mapped' if summary['original']['platform']=='jetson' else 'copy')]
    for field in ('phase','DPC_x'):
        arrays=[np.load(R/mode/p/(field+'.npy'),mmap_mode='r')[514:1514,514:1514] for p in selected]
        lo,hi=np.percentile(arrays[0],[1,99]);fig,ax=plt.subplots(1,len(arrays)+1,figsize=(4*(len(arrays)+1),4),constrained_layout=True)
        for i,a in enumerate(arrays):
            im=ax[i].imshow(a,cmap='viridis',vmin=lo,vmax=hi,origin='upper');ax[i].set_title(selected[i].replace('cli_'+mode+'_',''));ax[i].set_xlabel('central ROI x / pixel');ax[i].set_ylabel('central ROI y / pixel');fig.colorbar(im,ax=ax[i],shrink=.75)
        delta=arrays[1].astype(np.float64)-arrays[0];limit=max(np.max(np.abs(delta)),1e-12);im=ax[-1].imshow(delta,cmap='RdBu_r',vmin=-limit,vmax=limit);ax[-1].set_title('native - original');fig.colorbar(im,ax=ax[-1],shrink=.75)
        fig.suptitle(mode.upper()+' '+field+' | full input center1000 x1000');fig.savefig(O/(mode+'_'+field+'_comparison.png'),dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5),constrained_layout=True)
    for i,p in enumerate(selected):
        a=np.load(R/mode/p/'phase.npy',mmap_mode='r');ax.plot(np.arange(724,1324),a[1174,714:1314],label=p,ls=('-','--',':')[i],linewidth=1.7)
    ax.set_xlabel('input x / pixel, fixed y=1184');ax.set_ylabel('phase / rad');ax.legend(fontsize=7);fig.savefig(O/(mode+'_phase_profile.png'),dpi=160);plt.close(fig)
(O/'field_comparison.json').write_text(json.dumps(comparisons,indent=2))
with (O/'performance.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert all(q['pass_gate'] for m in comparisons.values() for p in m.values() for field,v in p.items() if field!='descriptors' for q in v.values()),'Output gate failed: preserve comparisons'
assert all(x for m in comparisons.values() for p in m.values() for x in p['descriptors'].values()),'Descriptor gate failed'
(O/'acceptance.json').write_text(json.dumps(dict(state='PASS',meaning='Engineering/numerical comparison with preserved GPU; not independent physical truth',rois=dict(center1000=[524,1524,524,1524],user_roi=[724,1324,924,1424]),phase_gate='max .002 +1e-5*maxabs; RMS .0002+1e-6*rms; rad'),indent=2))
# Light receipts only. Full .npy outputs and input datasets stay on each device.
with tarfile.open(R/'evidence_light.tgz','w:gz') as tar:
    for p in R.rglob('*'):
        if not p.is_file() or 'inputs' in p.parts or 'baseline' in p.parts:continue
        rel=p.relative_to(R)
        if rel.parts[0]=='release':
            if rel.parts[1]!='docs' or not p.name.startswith('build_'):continue
        if p.suffix in ('.json','.csv','.log','.png') and p.stat().st_size<20_000_000:tar.add(p,arcname=rel)
print('FULL_OUTPUTS_PASS',flush=True)
