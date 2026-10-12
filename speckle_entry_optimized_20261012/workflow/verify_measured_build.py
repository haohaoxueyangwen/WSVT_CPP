from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1];receipt=json.loads((root/'evidence/build_receipt.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(Path(p))==value for p,value in receipt['binaries'].items()),'Measured library changed'
for rel,value in receipt['source_sha256'].items():
    if rel.startswith(('cpu_cpp/source/','gpu_common/')) or rel=='tools/native_api.py':
        assert sha(root/rel)==value,rel
check=dict(measured_binaries_unchanged=True,tested_compute_sources_unchanged=True,
    original_libraries_retained=True,verification_scope='Recorded original/optimized binaries and compiled sources; packaging helpers may differ.')
(root/'evidence/final_verification.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check))
