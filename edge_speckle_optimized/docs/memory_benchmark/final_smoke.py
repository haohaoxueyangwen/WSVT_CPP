"""Verify final public API/defaults and optional configurations on BB11 crops."""
import gc,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch,tifffile
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'package'))
from speckle_edge import RunConfig,create_session,__version__
import run_experiment as exp
torch.set_num_threads(2);exp.idle();assert __version__=='0.1.1'
out=[]
for mode in ('wsvt','wxst'):
    arrays=[]
    for tag in ('sample','ref'):
        a=np.stack([tifffile.imread(exp.DATA/(tag+'_1_%05d.tif'%j))[896:1153,896:1159] for j in range(1,101 if mode=='wsvt' else 2)])
        arrays.append(np.ascontiguousarray(a if mode=='wsvt' else a[0],dtype='uint16' if mode=='wsvt' else 'float32'))
    s,r=arrays
    configs=[RunConfig(mode,pixel_m=1.625e-6,distance_m=.4845,energy_ev=12400.)]
    if mode=='wsvt':
        for filename in ('wsvt_speed.json','wsvt_memory.json'):configs.append(RunConfig(**json.loads((R/'package/configs'/filename).read_text())))
    with exp.session(mode,s.shape[-2:],'original') as old:
        old.prepare_reference(r);expected,_=old.solve(s)
        for config in configs:
            with create_session(config,s.shape[-2:],str(s.dtype)) as q:
                q.prepare_reference(r);actual,info=q.solve(s);checks=exp.equal_outputs(expected,actual);features=exp.equal_features(old,q)
                held={k:v.copy() for k,v in actual.items()};actual['phase'][0,0]+=1;held['phase'][0,0]+=1
                other,_=q.solve(r)
                assert all(np.array_equal(actual[k],held[k]) and not np.shares_memory(actual[k],other[k]) for k in actual)
                q.close();gc.collect();assert all(np.array_equal(actual[k],held[k]) for k in actual)
                out.append(dict(mode=mode,version=__version__,configuration=config.to_dict(),outputs=checks,features=features,ownership_pass=True,numpy_base_types={k:type(v.base).__name__ for k,v in actual.items()}))
exp.write(R/'final_smoke.json',out)
print('Final public API/defaults + both WSVT presets: exact old-GPU outputs/features and independent host ownership passed.')
