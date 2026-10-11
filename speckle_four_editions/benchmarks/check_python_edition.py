from pathlib import Path
import json,subprocess,sys
import numpy as np
R=Path(__file__).resolve().parent;P=R/'python_smoke_inputs';P.mkdir(exist_ok=False)
receipts={}
for mode in ('wxst','wsvt'):
    for n in ('sample','reference'):np.save(P/(mode+'_'+n+'.npy'),np.load(R/'inputs'/(mode+'_'+n+'.npy'),mmap_mode='r')[...,800:928,800:928])
    command=[sys.executable,str(R/'release/python_reference/run_reference.py'),'--mode',mode,'--sample',str(P/(mode+'_sample.npy')),'--reference',str(P/(mode+'_reference.npy')),'--out',str(R/('python_smoke_'+mode)),'--pixel-m','1.625e-6','--distance-m','.4845','--energy-ev','12400','--cores','2','--groups','2']
    with (R/('python_smoke_'+mode+'.log')).open('w') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    p.check_returncode();o=R/('python_smoke_'+mode);report=json.loads((o/'report.json').read_text());receipts[mode]=report
    for n in report['fields']:assert np.isfinite(np.load(o/(n+'.npy'))).all(),n
(R/'python_edition_smoke.json').write_text(json.dumps(receipts,indent=2));print('PYTHON_EDITION_SMOKE_PASS')
