# Two-platform native GPU results

BB11, full2048x2048. WSVT100 uint16 frames; WXST first float32 pair,7x7 template. Half10/coarse5/fine4,L2,I1,cut1,no dark-field,GPU spline. Reference rebuilt each pair; one warmup/five formal; mean ± sample SD.

Server: one RTX5880 Ada48GB, single EPYC9J14 (96physical cores/192logical). Orin: AGX Orin64GB shared DRAM,12ARM cores, MAXN unchanged. No competing compute jobs. GPU warmup/power history was not standardized with forced clock locking.

## Recommended native CLI deployments

| Device | Mode/policy | Solver s | Seven-output API s | Explicit device allocations GiB | Sampled process RSS GiB |
|---|---|---|---|---|---|
| server | wsvt speed/copy | 0.160533 ± 0.002983 | 0.164836 ± 0.002987 | 6.478 | 1.781 |
| server | wsvt low-memory/copy | 0.170445 ± 0.004298 | 0.174745 ± 0.004298 | 4.258 | 1.781 |
| server | wxst speed/copy | 0.056215 ± 0.001813 | 0.064156 ± 0.001812 | 1.595 | 0.250 |
| orin | wsvt speed/mapped | 1.190861 ± 0.000737 | 1.194558 ± 0.000764 | 5.696 | 1.731 |
| orin | wsvt low-memory/mapped | 1.399078 ± 0.000287 | 1.402875 ± 0.000356 | 3.476 | 1.731 |
| orin | wxst speed/copy | 0.595528 ± 0.000923 | 0.610231 ± 0.003863 | 1.595 | 0.200 |

Solver includes reference rebuild and sample compute with synchronization; API also includes return of seven fields. Native CLI reads input once, creates persistent pinned input/output/session buffers and reuses outputs. Constructor, pinned allocation/input loading and file saving are outside hot-path clocks and individually recorded. API is not end-to-end file latency.

## Original and same-host-policy controls

| Device/mode | Original pageable solver/API s | Native pageable solver/API s | Original pinned solver/API s | Native pinned solver/API s |
|---|---|---|---|---|
| server/wsvt | 0.198617 / 0.222639 | 0.197808 / 0.219982 | 0.160661 / 0.195508 | 0.157291 / 0.179332 |
| server/wxst | 0.047337 / 0.077971 | 0.044788 / 0.070803 | 0.046924 / 0.073617 | 0.051907 / 0.064700 |
| orin/wsvt | 1.398674 / 1.433283 | 1.384184 / 1.417337 | 1.248949 / 1.283011 | 1.266008 / 1.302438 |
| orin/wxst | 0.606501 / 0.653061 | 0.597776 / 0.639415 | 0.604679 / 0.649338 | 0.595222 / 0.636490 |

On pageable inputs, native compute changed little: WSVT0.1986→0.1978s server and1.3987→1.3842s Orin; WXST0.0473→0.0448s and0.6065→0.5978s. Pinning supplies much of the desktop WSVT transport benefit. On Orin, native WSVT mapped input improves1.2660s pinned-copy solver to1.1939s in the optional NumPy adapter; the standalone CLI confirms1.1909s. WXST mapping gave no material improvement.

Native server WXST CLI solver0.0562s was slower than the original pageable0.0473s in this batch; its ready-buffer API0.0642s was faster than0.0780s. Retain this result; do not claim universal kernel acceleration from C++ scheduling. Variation with buffer policy and GPU state requires profiling before assigning a cause.

## Memory accounting

Desktop common NVML process peaks: original WSVT6.988GiB, native speed6.836GiB, low-memory4.617GiB; original WXST2.111GiB, native1.955GiB. The WSVT low-memory policy cuts native explicit device allocations6.478→4.258GiB (~34.3%) at0.1605→0.1704s CLI solver cost (~6.2%).

Orin explicit device allocations for mapped WSVT speed/low-memory are5.696/3.476GiB (~39.0% reduction); CLI solver1.1909→1.3991s (~17.5% cost). These are CUDA allocations backed by shared DRAM, not independent VRAM. Corresponding sampled system-used peaks were16.001/14.030GiB, with RSS1.731GiB for both. Original pageable WSVT had system-used peak16.632GiB and RSS2.922GiB. System values include background/cache, so do not equate those differences to exact application savings. RSS and CUDA/shared-DRAM metrics overlap and must not be added.

Allocator accounting differs: old Torch max allocated versus native explicitly tracked allocations is not a total physical-peak equivalence. Use NVML for compatible desktop physical comparison. Orin has no independent VRAM counter; keep explicit allocations, RSS, system RAM and tegrastats separate. Samples25ms (tegrastats200ms) miss shorter peaks. Adapter memory formal stage and CLI whole-lifetime sampling have different scopes; all scopes retained in report.json.

## Reconstruction and reproducibility

All seven fields checked over full output, input center1000x1000 and fixed user ROI x724:1324,y924:1424 (equivalent to x200:800,y400:900 in the central image). Six nonphase fields and available descriptors matched exactly. Phase maximum/RMS differences (rad): server WSVT1.526e-4/2.054e-5, server WXST1.526e-4/1.736e-5; Orin WSVT1.068e-4/1.134e-5, Orin WXST9.155e-5/1.020e-5. All retained policies passed the predeclared gates in autoresearch.md; no thresholds were relaxed. These comparisons address preservation against the old implementation, not independent physical correctness.

Images: [server WSVT](results/server/analysis/wsvt_phase_comparison.png), [server WXST](results/server/analysis/wxst_phase_comparison.png), [Orin WSVT](results/orin/analysis/wsvt_phase_comparison.png), [Orin WXST](results/orin/analysis/wxst_phase_comparison.png). DPC and fixed profiles accompany these files. The difference colormaps intentionally magnify small phase changes.

Both new native builds have no libtorch/libpython linkage. GPU kernels/subpixel code remain frozen; native cuFFT/reduction accounts for small phase differences. CPU snapshots compiled on both hosts with FFTW; Python WSVT/WXST passed128² runner checks. Full new CPU timing was not repeated because CPU algorithms were unchanged.

Raw five-repeat CSV, individual stage/time records, warmup results, hashes, hardware/load observations, memory traces and full-field checks are retained under results/. Full arrays/datasets remain on hosts, outside GitHub. Failed initial build/helper attempts are retained as logs.

Remote server: /home/yrc/vscodespace/research/WSVT_project/bench_results/native_four_editions_20261011

Remote Orin: /home/xray/Documents/WSVT_GPU_orin_20261010/native_four_editions_20261011
