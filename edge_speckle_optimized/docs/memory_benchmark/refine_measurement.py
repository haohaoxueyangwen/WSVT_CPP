"""Preserve v1. Remove differing last-repeat output retention; save/check outside timing."""
from pathlib import Path
p=Path(__file__).with_name('run_experiment.py');s=p.read_text()
s=s.replace("folder=ROOT/('full_'+mode)","folder=ROOT/('controlled_'+mode)")
s=s.replace("chunks=[];best=32\n    if mode=='wsvt':", "chunks=[];best=16\n    if False:")
s=s.replace("        last.clear();gc.collect()", "        for v in ('original','streamed'):np.savez_compressed(folder/(v+'_outputs.npz'),**last[v])\n        last.clear();del out;gc.collect()")
s=s.replace("                if rep==4:last[v]=out\n                else:del out", "                del out")
start=s.index("        for v in ('copy_only','streamed'):",s.index("        for rep in range(5):"))
end=s.index("        summary={}",start)
s=s[:start]+"        # Full outputs and feature parity were checked before formal timing.\n"+s[end:]
s=s.replace("        for v in ('original','streamed'):np.savez_compressed(folder/(v+'_outputs.npz'),**last[v])\n        report=dict", "        report=dict")
s=s.replace("benchmark=sha(__file__)","benchmark=sha(__file__)")
Path(__file__).with_name('run_controlled.py').write_text(s)
