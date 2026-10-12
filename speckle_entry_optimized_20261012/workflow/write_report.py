from pathlib import Path
import json,datetime
root=Path(__file__).resolve().parents[1];e=root/'evidence';docs=root/'docs';docs.mkdir(exist_ok=True)
a=json.loads((e/'audit.json').read_text());rr={(r['mode'],r['backend']):r for r in a['rows']}
lines=['# 测速入口优化与同条件复测，2026-10-12','',
'WSVT 的完整入口耗时降至 CPU 约 5.95 s、原生 GPU 约 0.469 s。旧代码、原二进制和既往结果保留；本轮在独立副本中实现。', '',
'## 固定的比较口径','',
'同机 AMD EPYC 9J14、RTX 5880 Ada 单卡。2048×2048，WSVT 的参考/样品各 100 帧，WXST 各一帧；连续 pageable float32 主机输入。CPU 主线程 32、相位线程 4；GPU 主机有限值校验维持单线程，样条与 FFT 在 GPU。搜索总半窗 10、粗层 5、细层 4，两层空间金字塔、cut1、迭代 1 次，WXST 7×7 模板，WSVT 无空间模板，关闭暗场。', '',
'每次新建求解对象，重新准备参考，返回独立拥有的七字段主机结果。完整 array API 包含初始化、输入检查、参考重建、求解及结果返回；不含文件读写、导入/JIT 或计时后的析构。每组预热 1 次、正式测量 5 次。未使用参考缓存、固定输出复用、uint16 快速入口或持久会话来替代本口径。', '',
'## 完整 array API 结果','',
'单位为秒，均值 ± 样本标准差。同一行的旧版和优化版均在本轮重新测量。','',
'| 任务 | 后端 | 旧入口 | 优化入口 | 耗时减少 |', '|---|---|---:|---:|---:|']
for mode in ('wsvt','wxst'):
    for base,new,label in [('cpu_cpp','cpu_optimized','CPU C++'),('native_gpu','native_optimized','原生 GPU C++')]:
        x,y=rr[mode,base],rr[mode,new];pct=100*(1-y['array_api_mean_s']/x['array_api_mean_s'])
        lines.append(f"| {mode.upper()} | {label} | {x['array_api_mean_s']:.6f} ± {x['array_api_sd_s']:.6f} | {y['array_api_mean_s']:.6f} ± {y['array_api_sd_s']:.6f} | {pct:.1f}% |")
lines += ['', 'WSVT 的收益清楚；WXST 改进较小，GPU WXST 均值差小于单次运行标准差，不能据此声称稳定的大幅提速。', '',
'## 时间减少在哪里','',
'CPU 新增同步只读借用接口，不再先生成原始 sample/reference 的 std::vector 完整副本。WSVT 直接从调用者输入计算均值、方差，将归一化值写入私有未初始化缓冲；所需归一化工作区仍存在。WXST 从借用视图直接提取细层模板，随后持有自己的粗层工作区，去掉入口、构造器和金字塔入口的重复原始输入副本。调用期间输入必须保持有效且不可写。原有拥有数据的构造函数仍保留。', '',
'GPU 将逐元素、含早退分支的有限值检查改为独立主机 C++ 整数指数位归约；memcpy 读取位模式避免别名违规。编译器生成 SIMD，仍只用一个主机线程、不分配临时布尔数组。NaN/Inf 检查保留，每次参考准备和样品求解重新检查，不缓存可能被修改的输入校验结果。CUDA 保持 --fmad=false。', '',
'WSVT 的诊断均值如下。各列范围不同，因此只用于分解开销，不作为主表加速比：','',
'| 指标 | 旧版（s） | 优化版（s） |', '|---|---:|---:|']
old=rr['wsvt','cpu_cpp']['detail_means'];new=rr['wsvt','cpu_optimized']['detail_means']
for label,key in [('CPU 原始输入复制 / 优化版 O(1) 视图建立','copy_s'),('CPU solver() 调用墙钟','solver_wall_s'),('CPU 程序自报内部 solver','reported_solver_s')]:
    lines.append(f"| {label} | {old[key]:.6f} | {new[key]:.6f} |")
old=rr['wsvt','native_gpu']['detail_means'];new=rr['wsvt','native_optimized']['detail_means']
for label,key in [('GPU 初始化','constructor_s'),('GPU 参考内部准备（含传入）','reference_s'),('GPU 样品内部求解（含传入）','sample_solver_s'),('GPU 结果下载','download_s')]:
    lines.append(f"| {label} | {old[key]:.6f} | {new[key]:.6f} |")
key1='host_validation_and_reference_dispatch_s';key2='host_validation_and_sample_dispatch_s'
lines.append(f"| GPU 主机检查及接口分发剩余时间（参考＋样品） | {old[key1]+old[key2]:.6f} | {new[key1]+new[key2]:.6f} |")
lines += ['', '剩余时间通过外层接口时间减去内部准备/求解/下载得到，包含主机检查和少量接口工作，并非单独测得的检查器时间。CPU 的 displace_time 包含前端阶段，不能再与 pyramid/wavelet 相加。', '',
'## 重构及资源检查','',
'128² 小图检查后进行正式全图复测。CPU 与 GPU 各自新旧版本的位移、两路 DPC、相位、透射率及负 SSD 分数七字段均逐位相同；四后端使用同一输入内容，运行前后输入 SHA256 未变。这里验证工程改动没有改变计算结果。有限值检查共通过 123 项，包括 NaN、±Inf、有限极值、次正规数、正负零、非对齐地址、SIMD 尾部以及已校验输入被修改后的再次检查。', '',
'| 任务 | 后端 | 旧进程 VmHWM（GiB） | 优化进程 VmHWM（GiB） |', '|---|---|---:|---:|']
for mode in ('wsvt','wxst'):
    for base,new,label in [('cpu_cpp','cpu_optimized','CPU C++'),('native_gpu','native_optimized','原生 GPU 主机')]:
        def gib(r):return int(r['process_memory']['VmHWM'].split()[0])/1024**2
        lines.append(f"| {mode.upper()} | {label} | {gib(rr[mode,base]):.3f} | {gib(rr[mode,new]):.3f} |")
lines += ['', 'VmHWM 是含输入、输出及运行库的整个进程高水位 RSS。WSVT 所需归一化工作区仍存在，所以本轮减少的是复制流量和入口时间，内存峰值基本不变。WXST 的少量 RSS 差异也不支持显著内存降低的结论。GPU 显式设备分配总量新旧一致：WSVT 7,794,150,532 B、WXST 1,712,243,220 B（含 FFT 工作区；不是 nvidia-smi 总显存峰值）。', '',
'每组开始前两次间隔至少 30 s 的检查确认空闲，执行期间持续检查 GPU 竞争及外部 CPU 重负载，40 次正式测量没有检出竞争。', '',
'## 文件与使用','',
'- [完整计时图](../evidence/matched_entry_speed.png)',
'- [WSVT 相位与共同剖面](../evidence/wsvt_phase_comparison.png)',
'- [WXST 相位与共同剖面](../evidence/wxst_phase_comparison.png)',
'- [原始计时 CSV](../evidence/raw_timing.csv) 与 [审计 JSON](../evidence/audit.json)',
'- [固定协议](../autoresearch.md)、[检查及测速脚本](../workflow/matched_entry.py)、[构建原始记录](../evidence/build_receipt.json)', '',
'原版是相邻 speckle_four_editions_20261011；本优化版在 speckle_entry_optimized_20261012。CPU 新入口为 cpu_cpp/cpu_driver.cpp，旧入口文本保留为 cpu_driver_original.cpp；GPU 共用源在 gpu_common，desktop/jetson 构建入口分别在 gpu_discrete/gpu_jetson。', '',
'本轮是服务器同条件复测；Orin 尚未用这一新增入口版本重新测量。图片使用共同中心 512² 裁区，原输出坐标 [758:1270,758:1270]；四张相位图共用旧 CPU 裁区第 1/99 百分位色限，数据本身未截断。共同剖面取全图中心行，不是各版本独立选峰线。']
(docs/'BENCHMARK_20261012.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
journal=[dict(time=now,step='filesystem_setup',status='retained_failure',note='Initial finite-check report write failed because empty evidence directory was not archived. Created directory; identical assertions then passed.'),
dict(time=now,hypothesis='H1_CPU_readonly_borrow',decision='keep',wsvt_api_old_s=rr['wsvt','cpu_cpp']['array_api_mean_s'],wsvt_api_new_s=rr['wsvt','cpu_optimized']['array_api_mean_s'],evidence='audit.json; formal_cpu_wsvt_check.json; formal_cpu_wxst_check.json'),
dict(time=now,hypothesis='H2_GPU_SIMD_finite',decision='keep',validation='123 cases, one host thread',wsvt_api_old_s=rr['wsvt','native_gpu']['array_api_mean_s'],wsvt_api_new_s=rr['wsvt','native_optimized']['array_api_mean_s'],evidence='audit.json; finite_checks.json; formal_gpu_wsvt_check.json; formal_gpu_wxst_check.json'),
dict(time=now,step='full_matched_retest',status='complete',formal_trials=40,all_seven_fields_bitwise_unchanged=True,competition_detected=False)]
(root/'autoresearch.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in journal))
(docs/'AGENT_HANDOFF_CURRENT.md').write_text('''# Entry optimization handoff, 2026-10-12

Completed CPU read-only borrowed array entry and native GPU SIMD finite checks;
40 formal matched-array trials plus smoke/invalid-input tests passed. See
BENCHMARK_20261012.md and ../evidence/audit.json. All prior originals remain.

Remote independent directory: bench_results/entry_optimized_20261012. Controller
completed and exited; no recurring automation was created or resumed. Source,
binaries and original load records retained. Full seven-field outputs remain
on server; lightweight metrics/crops/figures are available locally.

Keep cold array API, internal solver, hot sessions and file IO distinct. CPU
32/4 is fixed. The borrowed profile does not silently support dark field,
initial estimation, cropping or adaptive raw-input experiments. Input lifetime
and immutability are caller responsibilities for the synchronous API.

Orin validation is pending for these new entry optimizations. Do not claim
server SIMD timings as Jetson results. No subpixel/SSD/FFT changes were made.
''',encoding='utf-8')
print(docs/'BENCHMARK_20261012.md')
