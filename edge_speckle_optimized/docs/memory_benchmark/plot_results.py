"""Same-scale BB11 image comparison; no masked outliers or rescaled outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':150,'savefig.dpi':170})
records={}
for mode in ('wsvt','wxst'):
    d=R/('controlled_'+mode);a=np.load(d/'original_outputs.npz');b=np.load(d/'streamed_outputs.npz')
    roi=np.s_[514:1514,514:1514] # Detector y/x524:1524 minus valid-field origin10.
    # A single row for each image field: original, candidate, difference.
    fig,axes=plt.subplots(2,3,figsize=(10,6.5),layout='constrained')
    fields=['phase','DPC_x'];label={'phase':'Phase (rad)','DPC_x':'DPC x (rad)'}
    for row,k in enumerate(fields):
        aa=a[k][roi];bb=b[k][roi];dif=bb.astype(np.float64)-aa.astype(np.float64)
        lo,hi=np.percentile(aa,[1,99]);maxabs=float(np.max(np.abs(dif)))
        for col,v in enumerate([aa,bb,dif]):
            ax=axes[row,col]
            if col==2:
                limit=maxabs if maxabs>0 else 1.
                im=ax.imshow(v,cmap='coolwarm',vmin=-limit,vmax=limit,extent=[524,1524,1524,524])
                ax.set_title('New - original; max abs = %.3g'%maxabs)
                cb=fig.colorbar(im,ax=ax,shrink=.8)
                if maxabs==0:cb.set_ticks([0])
            else:
                im=ax.imshow(v,cmap='viridis' if k=='phase' else 'RdBu_r',vmin=lo,vmax=hi,extent=[524,1524,1524,524]);ax.set_title(('Original GPU' if col==0 else 'Memory-optimized GPU')+'\n'+label[k]);fig.colorbar(im,ax=ax,shrink=.8)
            ax.set_xlabel('Detector x (px)');ax.set_ylabel('Detector y (px)')
    fig.suptitle(mode.upper()+' | BB11 center 1000 x 1000 | same display scales\n2048 x 2048 input; half10; original subpixel rules',fontsize=13)
    fig.savefig(F/(mode+'_center_comparison.png'));plt.close(fig)
    # Full valid-field map is supplied too, so the display ROI does not hide diagnostics.
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,k in zip(axes,('phase','displace_x','displace_y')):
        v=b[k];lo,hi=np.percentile(v,[1,99]);im=ax.imshow(v,cmap='viridis' if k=='phase' else 'RdBu_r',vmin=lo,vmax=hi,extent=[10,2038,2038,10]);ax.set_title({'phase':'Phase (rad)','displace_x':'Displacement x (px)','displace_y':'Displacement y (px)'}[k]);fig.colorbar(im,ax=ax,shrink=.8)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((524,524),1000,1000,fill=False,edgecolor='white',linewidth=1.4,linestyle='--'))
        ax.set_xlabel('Detector x (px)');ax.set_ylabel('Detector y (px)')
    fig.suptitle(mode.upper()+' optimized | full valid field2028 x 2028; white box = displayed ROI')
    fig.savefig(F/(mode+'_full_reconstruction.png'));plt.close(fig)
    records[mode]=dict(display_roi_detector_yx=[524,1524,524,1524],field_origin_yx=[10,10],fields={k:dict(exact=bool(np.array_equal(a[k],b[k])),full_max_abs=float(np.max(np.abs(a[k].astype(np.float64)-b[k])))) for k in a.files},display_clip_percentiles=[1,99],display_clip_only=True)
(F/'image_checks.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
