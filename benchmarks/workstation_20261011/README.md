# 四卡服务器上的当前 CUDA/C++ GPU 工程版实测

测量日期：2026-10-11。等待四卡训练结束后，两次相隔约50秒检查确认GPU无计算任务、CPU负载接近零，随后顺序执行测试。使用**GPU0单卡**；没有用四张卡同时处理同一幅图像。

硬件为单路AMD EPYC 9J14（96物理核心、192逻辑线程）、RTX 5880 Ada 48GB。CPU正式对照为原保存的FASTWSVT-cpp/FASTWXST-cpp CLI，32主线程、4相位线程；GPU工程版宿主线程设置为2。GPU使用CUDA 12.8.93、SM89，保留`--fmad=false`。Torch 2.1.2+cu118负责控制和FFT，匹配等主体由原生CUDA核完成。算法源文件未改动，仅通过CMake命令指定架构。

## 结果

真实BB11输入为2048×2048 uint16；WSVT使用100帧扫描，WXST使用第一对图像、7×7空间模板。两种模式均为总搜索半窗10、粗层半窗5、细层半窗4、两层空间金字塔、1次迭代、cut1，关闭暗场。WSVT不启用运算重排、不拼接空间邻域，GPU保留uint16输入。每后端预热1次、正式测量5次；下表为均值±样本标准差。

| 任务 | 原C++ CPU CLI求解器 / s | 当前GPU每对求解器 / s | 求解器加速比 | GPU返回七字段 / s |
|---|---:|---:|---:|---:|
| WSVT，2048²×100帧 | 6.5452±0.0658 | 0.20921±0.00449 | 31.29× | 0.22665±0.00440 |
| WXST，2048²、单对 | 2.7365±0.0114 | 0.04780±0.00107 | 57.25× | 0.06729±0.00138 |

GPU每次都重新建立参考描述子；上述加速比没有利用参考缓存。参考准备均值分别为86.02ms（WSVT）、4.78ms（WXST）；样品求解分别为123.19ms、43.02ms。返回七字段还包含将位移、DPC、相位、透射率与匹配分数下载为具有独立所有权的NumPy数组。

CPU求解器列来自原CLI的`solver_total_s`；GPU求解器列为参考准备墙钟时间加样品求解墙钟时间，包括上传和同步。两者均排除磁盘读写与冷启动。GPU返回结果列是内存API延迟，不能与CPU包含TIFF读取、进程启动及HDF5保存的CLI端到端时间直接计算加速比。CPU原生CLI文件处理耗时及阶段原始值另见原始CSV。

## 时间花在哪里

GPU样品阶段的CUDA事件计时均值如下，不含参考准备和结果下载；融合WXST前端的开销已包含在匹配阶段。

| 样品阶段 | WSVT / ms | WXST / ms |
|---|---:|---:|
| 输入写入/上传 | 51.69 | 1.11 |
| 样品前端 | 24.95 | 0.027 |
| 粗层匹配 | 10.01 | 9.00 |
| 样条位移传递 | 2.37 | 2.53 |
| 细层匹配 | 32.37 | 28.54 |
| 相位积分 | 1.18 | 1.19 |

WSVT的大图像栈上传已经是显著开销；这是独立显卡服务器上实际测到的时间，Jetson共用物理DRAM的访问方式不同。WXST主要时间仍在粗细匹配。持久GPU工作区约6.48GiB（WSVT）和1.59GiB（WXST）；这些是已分配工作区的字节和，**不是进程峰值显存测量**。

## 重构检查与边界

两种小图检查均通过原声明阈值。全尺寸正式输出七字段均具有预期形状、有限值和非零方差；200张输入图像的SHA256全部与原数据清单一致。

正式GPU输出与原CLI正式run_0的比较仍保留局部差异：WSVT水平/垂直位移RMSE约0.00827/0.00797px，有13个位置任一方向差异超过1px；WXST约0.07653/0.07444px，有708个此类位置。原声明的全图位移RMSE<0.005px阈值在两种模式下均未通过。相位NRMSE分别为3.89×10^-5、6.88×10^-4，透射率一致。原有边界亚像素拟合规则保持不变，这些计时作为当前工程版的性能结果，不能写成全图数值等价或独立物理精度验证。中心1000²及历史原图ROI的差异另保存在`summary.json`。

首次独立CPU共享库嵌入Torch控制进程的计时明显较慢：WSVT16.727s、WXST3.052s。其数据和失败阈值原样保留，但不用于上表的加速比；正式对照改用哈希验证过的原CLI并重新做1次预热、5次测量。进一步检查证实，设置Torch宿主线程为2后，该共享库的初始OpenMP默认上限也是2，显式设置后才成为32；部分前端循环依赖这一默认值。嵌入对照需要额外隔离并记录OpenMP环境，不能把其慢速直接归结为CPU硬件能力。这里没有用单次环境探针确定各项开销的全部成因。

## 文件与复现

服务器结果目录：`/home/yrc/vscodespace/research/WSVT_project/bench_results/edge_gpu_workstation_20261011`。

- `summary.json`、`summary.csv`：审计后的主要数据。
- `full_wsvt/`、`full_wxst/`：GPU/共享库原始每次计时、差异检查及正式七字段输出。
- `cli_wsvt/`、`cli_wxst/`：原C++ CLI的params、preflight、timing、aggregate、report、build provenance与全部正式输出。
- `build_receipt.json`：工具链、实际编译选项、源码/二进制哈希。
- `benchmark.py`、`cpu_cli.sh`、`finalize.py`：精确执行及审计脚本。

原CPU CLI哈希：`cb17c615b3a1a7e12d47f2206348dcd1f1b0a91adb9977bbeb5b317256a9b5f3`。
GPU源包哈希：`348b068d4354e67c9a6bb7ff62da1d372d3b96f6342aade54626281fc587408d`。

GPU构建在独立目录执行：

```bash
cmake -S package -B gpu_build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DEDGE_CUDA_ARCH=89 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu_build --target edge_kernels edge_spline -j4
```

测试结束时服务器再次确认无GPU计算进程；自动等待检查已暂停。原论文、原项目源码、原基准任务均未覆盖。
