from pathlib import Path
import hashlib,json,os,statistics,subprocess,time,zipfile
R=Path(__file__).resolve().parent
result=json.loads((R/'comparison.json').read_text());reports=json.loads((R/'summary.json').read_text())
def fmt(profile,k):
    v=reports[profile]['summary'][k];return '%.6f +/- %.6f'%(v['mean'],v['sample_sd'])
table=['| Profile | Pair solver (s) | Seven-field API (s) | Persistent buffers (GiB) | CUDA allocated peak (GiB) | Peak reserved (GiB) | Lifetime RSS peak (GiB) |','|---|---:|---:|---:|---:|---:|---:|']
for p in ('original','speed','low_memory'):
    m=reports[p]['memory'];table.append('| %s | %s | %s | %.4f | %.4f | %.4f | %.4f |'%(p,fmt(p,'solver_s'),fmt(p,'api_s'),m['persistent_device_bytes']/2**30,m['cuda_peak_allocated_bytes']/2**30,m['cuda_peak_reserved_bytes']/2**30,m['process_lifetime_peak_rss_bytes']/2**30))
base=reports['original'];low=reports['low_memory'];speed=reports['speed']
peak_reduction=100*(1-low['memory']['cuda_peak_allocated_bytes']/base['memory']['cuda_peak_allocated_bytes'])
resident_reduction=100*(1-low['memory']['persistent_device_bytes']/base['memory']['persistent_device_bytes'])
solver_penalty=100*(low['summary']['solver_s']['mean']/base['summary']['solver_s']['mean']-1)
api_delta_ms=1000*(base['summary']['api_s']['mean']-speed['summary']['api_s']['mean'])
stages=['| Profile | Reference preparation (s) | Sample feature GPU event (s) | Coarse match GPU event (s) | Fine match GPU event (s) |','|---|---:|---:|---:|---:|']
for p in ('original','speed','low_memory'):
    rr=reports[p]['records'];stages.append('| %s | %.6f | %.6f | %.6f | %.6f |'%(p,reports[p]['summary']['reference_s']['mean'],*[statistics.mean(v['detail']['gpu_event_s'][k] for v in rr) for k in ('sample_front','coarse_match','fine_match')]))
text='''# Orin WSVT memory/output-copy benchmark, 2026-10-11

## Measured result

'''+ '\n'.join(table)+'''

One warmup and five measured pairs per independent process, means +/- sample standard deviations. BB11 lens, 100 x 2048 x 2048 uint16. Every timed pair rebuilds the reference. Original0.1.0 and optimized0.1.1 use the same subpixel, SSD, spline, phase and temporal/spatial operator order.

Speed configuration (materialized workspace, independent host tensor outputs without an extra NumPy copy) preserves solver latency, with a small %.3f ms reduction in API latency. It is not a substantive solver acceleration.

Low-memory configuration (streamed workspace, spatial channel chunk16) reduces persistent device buffers by %.2f%% and CUDA allocated peak by %.2f%%, while pair solver latency increases %.2f%%. On this Orin, use speed mode when latency is the priority; use low-memory mode when DRAM headroom matters. This benchmark does not change either preset to conceal the tradeoff.

## Stage evidence

'''%(api_delta_ms,resident_reduction,peak_reduction,solver_penalty)+'\n'.join(stages)+'''

The extra low-memory time appears primarily in reference/sample feature generation. Coarse/fine matching timings remain close. Kernel occupancy, instruction throughput and bandwidth were not profiled, so these measurements do not establish a particular hardware cause.

## Memory and timing scopes

CUDA allocator peaks were reset after warmup and measured across five complete pairs. Persistent buffers are the session's explicit device tensors; peaks also include transient FFT/pointwise workspace. Reserved peak is the PyTorch allocator reservation, not an additional allocation to sum with allocated peak.

Linux ru_maxrss is the process lifetime RSS through the end of formal solves: imports, input decoding, session initialization, warmup and measured runs, excluding subsequent descriptor hashing and file saving. Host RSS does not show a reduction: values span about 2.93-3.05 GiB. Device allocations and host RSS share physical DRAM on Jetson and must not be summed as independent hardware memories.

25 ms sampled process/system RAM and 200 ms tegrastats logs are included. Sampled peaks are lower bounds, and system MemTotal-MemAvailable includes other processes and page-cache effects. The system-level snapshot is not a precise per-process memory attribution.

Solver time: reference preparation + sample solve, including uploads and final CUDA synchronization, excluding validation/download, session construction and disk I/O. API time: already-decoded contiguous host arrays to seven independent writable NumPy arrays, including input validation, reference rebuild and download; excluding construction and disk I/O. Read/decode and output-save times are recorded separately, not mixed into API time.

## Numerical and visual comparison

All seven full-field outputs are exactly equal to the original GPU outputs in both optimized presets. Fine/coarse reference and sample descriptors have identical SHA256 values. Odd/even crops, uint16/float32, constant-zero-standard-deviation and ownership-after-reuse/close checks also passed. These checks establish preservation of the original GPU implementation, not independent physical accuracy.

Figures show all three phases over the full valid2028 x2028 field, central1000 x1000 phase/DPCx images with identical scales, differences, and fixed detector-y1024 profiles. 1-99% clipping is only for display; full comparisons use all pixels.

## Device/build/provenance

Jetson AGX Orin, shared ~61 GiB CPU/GPU DRAM, CC8.7/16 SM, CUDA11.4, GCC9.4, PyTorch2.0.0+nv23.05. MAXN mode0; existing power/clock policy preserved. No jetson_clocks/nvpmodel mutation. Actual native flags include -O3 --std=c++17 --fmad=false -arch=sm_87. CMake3.16 ignores the generic CMAKE_CUDA_ARCHITECTURES variable; the explicit -arch=sm_87 target flag is present and was checked.

Sample/reference canonical uint16 hashes match the server TIFF inputs. Source archive SHA256:8577c1006d9b9579e66c8c77c2bd476c8da21cdbe44a868377cfa58f9c93b1c9. Native binaries were rebuilt for ARM/SM87 in this independent directory; baseline binaries were copied without rebuilding. Original source/native hashes remain unchanged after the runs.

Original: /home/xray/Documents/WSVT_GPU_orin_20261010/edge_speckle
New deployment/results: /home/xray/Documents/WSVT_GPU_orin_20261010/memory_opt_20261011
Speed/low-memory preset examples: package/configs/wsvt_speed.json and wsvt_memory.json.

Full .npy arrays remain on Orin. Lightweight evidence bundle excludes input arrays, reconstruction arrays, source archives, build objects and credentials.

## Reproduction

Review autoresearch.md and source/build receipts. Preserve existing results. In a fresh independent result directory, deploy the same package, then run python3 -u run_orin.py batch and python3 analyze_orin.py. The runner intentionally refuses to overwrite original/speed/low_memory result folders. No test was run concurrently with a competing user GPU job in the inspected state.
'''
(R/'README.md').write_text(text)
(R/'AGENT_HANDOFF_CURRENT.md').write_text('Independent Orin comparison complete. Original/source/binaries preserved. See README.md, comparison.json/csv, smoke.json, summary.json, native build receipt and figures. No subpixel change. Speed mode retains old solver latency; low-memory mode trades ~15.4% longer solver time for ~33.8% lower allocated peak. No production default changed by this experiment.\n')
(R/'PAPER_ROADMAP.md').write_text('Incremental evidence only: Orin GPU engineering benchmark for original/speed/low-memory WSVT. Report separate solver/API timings and CUDA allocator vs RSS scopes. Exact old-GPU preservation is not physical truth. Do not insert these timings into EPYC/Ada tables; keep hardware-specific. No original manuscript has been edited.\n')
(R/'AI_PROJECT_BRIEF.md').write_text('Completed user-authorized BB11 uint16100x2048x2048 benchmark, half10/coarse5/fine4/L2/I1/cut1/noDF/GPU spline/host2. Original0.1.0 untouched; optimized0.1.1 isolated. Seven fields and four descriptor hashes exact. Power/clock settings unchanged. Further low-memory kernel tuning requires a separate bounded experiment; current preset measurements remain frozen.\n')
(R/'autoresearch.jsonl').write_text(json.dumps(dict(timestamp=time.time(),experiment='orin_wsvt_original_speed_low_memory',status='complete',source_unchanged=True,native_unchanged=True,result=result,notes='No algorithm/operator-order/subpixel changes; no clocks changed.'))+'\n')
end=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),load=os.getloadavg(),meminfo=Path('/proc/meminfo').read_text(),processes=subprocess.check_output(['ps','-eo','pid,pcpu,comm'],text=True),idle_tegrastats=subprocess.run(['timeout','3','tegrastats','--interval','1000'],stdout=subprocess.PIPE,text=True).stdout)
(R/'postflight.json').write_text(json.dumps(end,indent=2))
flags=(R/'package/build/CMakeFiles/edge_kernels.dir/flags.make').read_text();assert '--fmad=false' in flags and '-arch=sm_87' in flags
(R/'actual_cuda_flags.txt').write_text(flags)
files=[p for p in R.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl','.csv','.log','.txt') and p.name not in ('evidence_manifest.json','pack.log')]
files+=list((R/'figures').glob('*'))
for profile in ('original','speed','low_memory'):files+=[R/profile/f for f in ('report.json','records.json','timing.csv','stage.json')]
files += [R/'package/speckle_edge/_native/build_info.json']
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(R/'evidence_manifest.json')
with zipfile.ZipFile(R/'orin_memory_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(R).as_posix())
print('PACK_COMPLETE',json.dumps(dict(zip_bytes=(R/'orin_memory_evidence.zip').stat().st_size,peak_reduction_percent=peak_reduction,solver_penalty_percent=solver_penalty)),flush=True)
