"""Same controlled protocol, final public native binary."""
from pathlib import Path
import run_controlled as measured
R=Path(__file__).resolve().parent
# Keep all previous outputs and trials. Only the fresh output directory changes.
source=(R/'run_controlled.py').read_text().replace("folder=ROOT/('controlled_'+mode)","folder=ROOT/('published_'+mode)")
exec(compile(source,str(__file__),'exec'),globals())
