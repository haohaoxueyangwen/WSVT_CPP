from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parent;P=R/'package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((P/'speckle_edge/_native/build_info.json').read_text())
receipt=dict(build=build,published_source_zip_sha256=sha(R/'published_source_0.1.1.zip'),
    measured_previous_library_sha256=sha(R/'measured_native/libedge_kernels.so'),published_library_sha256=sha(P/'speckle_edge/_native/libedge_kernels.so'),
    benchmark_sources_sha256={n:sha(R/n) for n in ('run_published.py','run_controlled.py','run_experiment.py')},
    interpretation='Different binary after final build: reran one warmup/five repeats. published_* is primary; all previous trials retained.')
for m in ('wxst','wsvt'):
    q=json.loads((R/('published_'+m)/'report.json').read_text());assert q['hashes']['new_library']==receipt['published_library_sha256']
    c=json.loads((R/('controlled_'+m)/'report.json').read_text())
    assert all(q['checks']['streamed']['outputs'][n]['exact'] and q['checks']['streamed']['outputs'][n]['sha256']==c['checks']['streamed']['outputs'][n]['sha256'] for n in q['checks']['streamed']['outputs'])
(R/'published_build_receipt.json').write_text(json.dumps(receipt,indent=2))
files=[R/n for n in ('published_build_receipt.json','published_build.log','published_smoke.log','final_smoke.json','run_published.py','published_wsvt.log','published_wxst.log')]
for m in ('wxst','wsvt'):files.extend(p for p in (R/('published_'+m)).iterdir() if p.suffix in ('.json','.csv'))
with zipfile.ZipFile(R/'published_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(R))
print('Final library, both full-output hashes and timing receipt verified; lightweight published evidence packed.')
