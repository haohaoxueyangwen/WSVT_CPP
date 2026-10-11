# WSVT / WXST：Jetson 工程整理版

当前维护入口集中在本目录，版本号 `0.1.0`。原 Python、FASTWSVT-cpp / FASTWXST-cpp、旧 GPU 原型、论文及已有实验文件均保留。这次整理改变工程组织、参数校验和记录方式，匹配、小波、样条和相位计算规则保持原样。

## 哪个版本用于什么

| 版本 | 用途 | 位置 |
|---|---|---|
| 原 Python、FastXST-C | 论文历史基线 | 原项目与服务器，参见 `docs/VERSIONS.md` |
| FASTWSVT-cpp / FASTWXST-cpp | 多核 CPU 基线 | `baselines/cpu/` 保存原服务器源码快照和独立适配器 |
| GPU v1 | 分阶段 CUDA 原型 | `baselines/legacy_wxst/wxst_orin.py` |
| GPU v2 | 模板、归一化、小波前端融合 | `baselines/legacy_wxst/wxst_orin_v2.py` |
| GPU v3 | 前端融合，候选代价缓存 | `baselines/legacy_wxst/wxst_orin_v3.py` |
| edge WXST | 当前工程默认：样品特征/搜索融合，GPU 样条 | `python3 -m speckle_edge wxst` |
| edge WSVT | 当前工程默认：原顺序，uint16 自动保留，GPU 样条 | `python3 -m speckle_edge wsvt` |
| WSVT reordered | 运算重排实验，默认关闭 | 必须显式指定 `--front reordered --allow-experimental` |

旧原型不自动补齐输出，也不作为默认入口。CPU 基线独立构建，当前统一 CLI 调用的是 edge GPU。完整区别、输出能力和验证范围见 [版本说明](docs/VERSIONS.md)。

## 目录

```text
speckle_edge/       Python API、两个方法入口、参数和结果契约
  _native/         本机生成的 CUDA/CPU 样条库与构建记录
native/            独立 CUDA 源码与 CPU 样条；不再引用父级实验目录
configs/           WXST、WSVT 分开的配置模板
tools/             构建、源码核对
tests/             参数契约测试和 Orin 整理前后对照
baselines/         历史 GPU 源码、原 C++ CPU 快照
benchmarks/        计时边界和复测规则
docs/              架构、版本、输出、验证和交接记录
runs/              自选新输出目录；不进入源码包
```

数据和大规模结果继续留在原路径，不重复搬进工程包。版本及证据索引见 `docs/versions.json`、`docs/evidence_registry.json`。

## Orin 构建

实测环境为 Ubuntu20.04 / aarch64、CUDA11.4、SM87、系统 Python3.8、NVIDIA PyTorch2.0.0+nv23.05、NumPy、g++9.4 / OpenMP。PyTorch 应使用适配 Jetson 的安装包，项目不会从 PyPI 自动安装通用 torch。

```bash
cd /home/xray/Documents/WSVT_GPU_orin_20261010/edge_speckle
/usr/bin/python3 tools/build.py --jobs 2
/usr/bin/python3 -m speckle_edge doctor
/usr/bin/python3 -m unittest discover -s tests -v
```

CMake 最低3.16；CUDA 固定 `-O3 --fmad=false -arch=sm_87`。构建记录保存在 `speckle_edge/_native/build_info.json`，包含实际编译器、CMake 配置、全部原生源码和二进制哈希。无需更改系统工具链或功耗配置。

## 两个入口

```bash
# 单对图像：sample/reference 均为 H×W。
/usr/bin/python3 -m speckle_edge wxst --input ../bb11_pair256.npz \
  --out runs/wxst_001 --pixel-m 1.625e-6 --distance-m .4845 --energy-ev 12400

# 多帧扫描：sample/reference 均为100×H×W。
/usr/bin/python3 -m speckle_edge wsvt --input ../edge_adaptation_20261011/cli_input_u16_128.npz \
  --out runs/wsvt_001 --pixel-m 1.625e-6 --distance-m .4845 --energy-ev 12400
```

这些几何值只用于复现已有工程测试；实际样品须填写自己的像素尺寸、有效传播距离和能量。配置模板中的几何字段为 `null`，必须填写或用 CLI 覆盖。模板不能冒充物理标定。

```bash
/usr/bin/python3 -m speckle_edge wxst --config configs/wxst.json --input YOUR_PAIR.npz \
  --out runs/NEW_RUN --pixel-m YOUR_PIXEL --distance-m YOUR_DISTANCE --energy-ev YOUR_ENERGY
```

固定算法配置：总半窗10、粗层半径5、细层半径4、两层空间金字塔、一次迭代、小波截断1；WXST 模板7×7 / 描述子58维，WSVT100帧 / 描述子63维且无空间模板。修改 JSON 中的数值不会自动生成新 CUDA 算法，不支持的配置会直接报错。

## 在代码中使用

```python
from speckle_edge import RunConfig, create_session
from speckle_edge.config import validate_arrays

config = RunConfig('wxst', pixel_m=1.625e-6, distance_m=.4845, energy_ev=12400.)
sample, reference, dtype = validate_arrays(config, sample, reference)
with create_session(config, sample.shape[-2:], dtype) as session:
    session.prepare_reference(reference)
    outputs, times = session.solve(sample)
```

会话复用输入、描述子、搜索及样条缓冲区；参考准备为显式操作。会话由创建线程独占，一个会话只允许一次在途求解。返回的 NumPy 结果拥有独立存储，后续求解不覆盖旧结果。不同工作线程应各持有一个会话，并评估并存的内存成本。

每个新结果目录包含 `reconstruction.npz`、`report.json` 和 `status.json`。七个字段、单位及原图坐标见 [结果契约](docs/OUTPUTS.md)；求解器、API 和文件 I/O 分开记录。现有结果目录拒绝覆盖，失败目录记录具体异常。

## 当前验证范围

工程回归验证整理前后的数值、缓冲复用和生命周期；结果见 [验证记录](docs/VALIDATION.md)。既有 CPU/GPU 局部差异的诊断继续保留。用户已确认关注区域可用，这支持继续工程整理；全图差异统计保持原始记录。没有采用“99%一致”作为恢复正确性的指标。

默认维护路线是 edge WXST 与 edge WSVT。后续优化先记录候选版本和固定协议，再测结果与速度；稳定入口不随实验代码自动改变。
