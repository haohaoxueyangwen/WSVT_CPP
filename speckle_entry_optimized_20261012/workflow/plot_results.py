from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];e=root/'evidence'
audit=json.loads((e/'audit.json').read_text());assert audit['all_gate_pass']
rows={(r['mode'],r['backend']):r for r in audit['rows']}
raw=list(csv.DictReader((e/'raw_timing.csv').open()))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False})
fig,axes=plt.subplots(2,2,figsize=(7.1,5.4),layout='constrained')
for i,mode in enumerate(('wsvt','wxst')):
    for j,(base,new,label) in enumerate((('cpu_cpp','cpu_optimized','CPU C++ 32 / phase 4'),('native_gpu','native_optimized','Native GPU, one card'))):
        ax=axes[i,j];rr=[rows[mode,b] for b in (base,new)]
        ax.bar([0,1],[r['array_api_mean_s'] for r in rr],yerr=[r['array_api_sd_s'] for r in rr],
               color=['#b3bdc9','#6893b8'],capsize=4,width=.55)
        for x,b in enumerate((base,new)):
            ys=[float(v['array_api_s']) for v in raw if v['mode']==mode and v['backend']==b]
            ax.scatter(x+np.linspace(-.11,.11,len(ys)),ys,color='#25364b',s=9,zorder=3)
            ax.text(x,max(ys)+.045*max(r['array_api_mean_s'] for r in rr),f"{rr[x]['array_api_mean_s']:.3f} s",ha='center',va='bottom')
        ax.set_xticks([0,1],['Original','Optimized']);ax.set_ylabel('Complete array API (s)')
        ax.set_ylim(0,1.30*max(float(v['array_api_s']) for v in raw if v['mode']==mode and v['backend'] in (base,new)))
        ax.set_title(f'{mode.upper()} | {label}',fontsize=9)
fig.suptitle('2048 x 2048 | float32 pageable inputs | fresh object + reference rebuild',fontsize=9)
fig.supxlabel('Five formal trials; bars: mean +/- sample SD; dots: each measured trial',fontsize=8)
for ext in ('png','svg','pdf'):fig.savefig(e/f'matched_entry_speed.{ext}',dpi=240)
plt.close(fig)
data=np.load(e/'comparison_rois.npz')
for mode in ('wsvt','wxst'):
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.1),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.35]})
    lo,hi=np.percentile(data[f'{mode}_cpu_cpp_phase'],[1,99])
    for row,(base,new,label) in enumerate((('cpu_cpp','cpu_optimized','CPU C++'),('native_gpu','native_optimized','Native GPU'))):
        aa=data[f'{mode}_{base}_phase'];bb=data[f'{mode}_{new}_phase']
        for col,(a,title) in enumerate(((aa,'Original'),(bb,'Optimized'))):
            im=axes[row,col].imshow(a,cmap='viridis',vmin=lo,vmax=hi,origin='upper',interpolation='nearest')
            axes[row,col].set_title(f'{label}: {title}',fontsize=8);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        fig.colorbar(im,ax=axes[row,:2],shrink=.75,label='Phase (rad)',fraction=.04,pad=.02)
        ax=axes[row,2];a=data[f'{mode}_{base}_phase_profile'];b=data[f'{mode}_{new}_phase_profile']
        ax.plot(a,color='#6893b8',lw=2,label='Original')
        ax.plot(b,color='#25364b',lw=1,ls=(0,(5,3)),label='Optimized')
        ax.set_xlabel('Output column (px)');ax.set_ylabel('Phase (rad)');ax.set_title('Same central row',fontsize=8);ax.legend(fontsize=7)
    fig.suptitle(f'{mode.upper()} | same central 512 x 512 crop and color scale',fontsize=9)
    fig.supxlabel('Display limits: original CPU phase 1st/99th percentiles; full-array differences = 0.',fontsize=7)
    for ext in ('png','svg','pdf'):fig.savefig(e/f'{mode}_phase_comparison.{ext}',dpi=240)
    plt.close(fig)
