"""Full-field parity and same-scale plots; display clipping never changes comparisons."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path(__file__).resolve().parent;F=R/'figures';F.mkdir(exist_ok=True)
PROFILES=('original','speed','low_memory');LABELS=('Original GPU 0.1.0','Optimized speed','Optimized low-memory')
FIELDS=('displace_y','displace_x','DPC_y','DPC_x','phase','transmission','match_score_neg_ssd')
reports=json.loads((R/'summary.json').read_text());checks={};telemetry={}
for profile in PROFILES:
    log=(R/(profile+'_tegrastats.log')).read_text();telemetry[profile]={}
    for k,pattern in [('gpu_util_percent',r'GR3D_FREQ (\d+)%'),('gpu_temperature_C',r'GPU@([\d.]+)C'),('gpu_clock_MHz',r'GR3D_FREQ \d+%@\[(\d+)'),('emc_util_percent',r'EMC_FREQ (\d+)%')]:
        values=[float(v) for v in re.findall(pattern,log)];telemetry[profile][k]=dict(min=min(values),max=max(values),mean=float(np.mean(values))) if values else None
    if profile=='original':continue
    checks[profile]={}
    for n in FIELDS:
        a=np.load(R/'original'/(n+'.npy'),mmap_mode='r');b=np.load(R/profile/(n+'.npy'),mmap_mode='r');assert a.shape==b.shape
        max_abs=0.;sumsq=0.;different=0;roi=np.s_[524:1524,524:1524] if n=='transmission' else np.s_[514:1514,514:1514]
        for j in range(0,a.shape[0],128):
            d=b[j:j+128].astype(np.float64)-a[j:j+128];max_abs=max(max_abs,float(np.max(np.abs(d))));sumsq+=float(np.sum(d*d));different+=int(np.count_nonzero(d))
        dr=b[roi].astype(np.float64)-a[roi]
        checks[profile][n]=dict(shape=list(a.shape),exact=different==0,different_pixels=different,full_max_abs=max_abs,full_rmse=float(np.sqrt(sumsq/a.size)),center1000_max_abs=float(np.max(np.abs(dr))),sha256=hashlib.sha256(memoryview(b).cast('B')).hexdigest())
    checks[profile]['descriptors_exact']={n:reports[profile]['features_sha256'][n]==reports['original']['features_sha256'][n] for n in reports['original']['features_sha256']}
assert all(v[n]['exact'] for v in checks.values() for n in FIELDS),checks
assert all(all(v['descriptors_exact'].values()) for v in checks.values()),checks
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'savefig.dpi':160})
roi=np.s_[514:1514,514:1514]
fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
for row,n in enumerate(('phase','DPC_x')):
    base=np.load(R/'original'/(n+'.npy'),mmap_mode='r')[roi];lo,hi=np.percentile(base,[1,99])
    for col,profile in enumerate(PROFILES):
        v=np.load(R/profile/(n+'.npy'),mmap_mode='r')[roi];ax=axes[row,col]
        im=ax.imshow(v,cmap='viridis' if n=='phase' else 'RdBu_r',vmin=lo,vmax=hi,extent=[524,1524,1524,524]);fig.colorbar(im,ax=ax,shrink=.75)
        ax.set_title(LABELS[col]+'\n'+('Phase (rad)' if n=='phase' else 'DPC x (rad)'));ax.set_xlabel('Detector x (px)');ax.set_ylabel('Detector y (px)')
fig.suptitle('Orin | BB11 WSVT | center 1000 x 1000 | identical display scales\n100 x 2048 x 2048 uint16; half10; GPU spline; original subpixel rules')
fig.savefig(F/'wsvt_three_versions_center.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
base=np.load(R/'original/phase.npy',mmap_mode='r');lo,hi=np.percentile(base,[1,99])
for col,profile in enumerate(PROFILES):
    ax=axes[col];v=np.load(R/profile/'phase.npy',mmap_mode='r');im=ax.imshow(v,cmap='viridis',vmin=lo,vmax=hi,extent=[10,2038,2038,10]);fig.colorbar(im,ax=ax,shrink=.8)
    ax.add_patch(Rectangle((524,524),1000,1000,fill=False,edgecolor='white',linewidth=1.6,linestyle='--'));ax.set_title(LABELS[col]+'\nPhase (rad)');ax.set_xlabel('Detector x (px)');ax.set_ylabel('Detector y (px)')
fig.suptitle('Full valid field 2028 x 2028 | white box: center ROI | 1-99% display clipping only')
fig.savefig(F/'wsvt_three_versions_full_phase.png');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(8.5,7),constrained_layout=True)
for row,n in enumerate(('phase','DPC_x')):
    for col,profile in enumerate(PROFILES[1:]):
        a=np.load(R/'original'/(n+'.npy'),mmap_mode='r')[roi];b=np.load(R/profile/(n+'.npy'),mmap_mode='r')[roi];d=b.astype(np.float64)-a
        ax=axes[row,col];maximum=float(np.max(np.abs(d)));im=ax.imshow(d,cmap='coolwarm',vmin=-max(maximum,1e-15),vmax=max(maximum,1e-15),extent=[524,1524,1524,524]);cb=fig.colorbar(im,ax=ax,shrink=.8)
        if maximum==0:cb.set_ticks([0])
        ax.set_title(LABELS[col+1]+' - original\n'+n+'; max |difference| = %.3g'%maximum);ax.set_xlabel('Detector x (px)');ax.set_ylabel('Detector y (px)')
fig.suptitle('Orin WSVT | center 1000 x 1000 difference maps')
fig.savefig(F/'wsvt_differences.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
for ax,n in zip(axes,('phase','DPC_x')):
    for profile,label,style in zip(PROFILES,LABELS,('k-','r--','b:')):
        v=np.load(R/profile/(n+'.npy'),mmap_mode='r');ax.plot(np.arange(524,1524),v[1024-10,514:1514],style,label=label,linewidth=1.6)
    ax.set_xlabel('Detector x (px), detector y = 1024');ax.set_ylabel('Phase (rad)' if n=='phase' else 'DPC x (rad)');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(n+' | all three traces coincide')
fig.savefig(F/'wsvt_profiles.png');plt.close(fig)
rows=[]
for profile in PROFILES:
    v=reports[profile];row=dict(profile=profile)
    for k in ('solver_s','api_s','reference_s','sample_s','download_s'):
        for t in ('mean','sample_sd'):row[k+'_'+t]=v['summary'][k][t]
    for k,value in v['memory'].items():
        if k.endswith('_bytes'):row[k+'_GiB']=value/2**30
    row['solver_speedup']=reports['original']['summary']['solver_s']['mean']/v['summary']['solver_s']['mean'];row['api_speedup']=reports['original']['summary']['api_s']['mean']/v['summary']['api_s']['mean'];rows.append(row)
with (R/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
original=R.parent/'edge_speckle';pre=json.loads((R/'preflight.json').read_text())
source_unchanged=all(hashlib.sha256((original/n).read_bytes()).hexdigest()==v['actual'] for n,v in pre['original_source_checks'].items())
binary_unchanged=all(hashlib.sha256((original/n).read_bytes()).hexdigest()==h for n,h in pre['original_native_sha256'].items())
assert source_unchanged and binary_unchanged
result=dict(summary=rows,checks=checks,telemetry=telemetry,original_source_unchanged=source_unchanged,original_native_unchanged=binary_unchanged,
    display=dict(center_roi_detector_yx=[524,1524,524,1524],valid_field_origin_yx=[10,10],clipping_percentiles=[1,99],clipping_only_for_display=True),
    numerical_scope='Comparison with the original GPU implementation; not an independent physical ground truth.')
(R/'comparison.json').write_text(json.dumps(result,indent=2));(F/'image_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(result['summary'],indent=2),flush=True)
