# 版本职责与迁移关系

这里使用版本的功能名称区分算法改动、执行方式与工程封装。`v1/v2/v3` 是已有 WXST GPU 原型标签；`0.1.0` 是本工程包版本。二者不是同一条数字序列。

| 版本 | 主要计算方式 | 参考/样品描述子 | 样条 | 完整七字段 | 维护状态 |
|---|---|---|---|---|---|
| 原 Python / FastXST-C | 原前端与多进程匹配 | 原数据流 | 原实现 | 依原输出协议 | 历史比较，不修改 |
| FASTWSVT-cpp / FASTWXST-cpp | HWD 描述子、多核 OpenMP、CPU SIMD | 预计算 | CPU | CPU 适配器统一为七字段 | 保留同机对照 |
| GPU v1 | 各前端阶段、SSD/亚像素分别启动 | 全部物化 | CPU SciPy | 原型缺透射率 | 冻结 |
| GPU v2 | 模板/归一化/小波融合 | 全部物化 | CPU SciPy | 原型缺透射率 | 冻结 |
| GPU v3 | v2＋候选代价缓存，拟合复用代价 | 全部物化 | CPU SciPy | 原型缺透射率 | 冻结；完整输出比较用 `legacy_complete.py` |
| edge WXST | 参考预计算；样品前端与搜索融合 | 参考 HWD，样品无需全图描述子 | 默认 GPU，CPU 可选 | 有 | 当前工程默认 |
| edge WSVT baseline | 100帧归一化→空间金字塔→各层时间小波 | 参考和样品 HWD | 默认 GPU，CPU 可选 | 有 | 当前工程默认；float32 / uint16 |
| edge WSVT reordered | 时间小波后建立描述子金字塔 | 63通道金字塔 | GPU/CPU | 有 | 显式实验，保留失败记录 |

CPU 的 OpenMP 来自运行库，匹配循环由本项目实现；SIMD 使用原工程的编译器向量化路径。Jetson CUDA 实现使用原生 kernel，Python 负责入口、会话和调度，不使用 `use_gpu` 字段推断 GPU 执行。

`edge WXST materialized` 是前端融合的对照开关，保留完整样品描述子；用于检查数据流和资源变化。它不另取一个含糊的“更优版本”编号。选择 CPU 样条仅切换位移传递后端，前端/搜索仍在 GPU。

## 原始来源

- Windows 原 C++ 项目：`D:/Yangrc/code/WSVT_project/WSVT_CPP`；与根项目是两个 Git 仓库。
- 原服务器 CPU 源码：`/home/yrc/vscodespace/research/WSVT_project/WSVT_CPP`。
- Orin 原型：`/home/xray/Documents/WSVT_GPU_orin_20261010`。
- Orin 工程候选：上述目录的 `edge_adaptation_20261011`。
- Orin 同机 CPU 对照：上述目录的 `cpu_compare_20261011`。

新包将可编译依赖移到 `native/`；历史源码复制到 `baselines/`。转换清单 `source_origins.json` 逐文件保存原始 SHA256、整理后 SHA256和明确的工程修改。没有删除实验历史，没有编辑论文文件，没有提交或重置原有脏工作区。

## 默认版本升级规则

1. 给候选注册唯一 `variant_id`，记录输入模型、参数和输出能力。
2. 用固定数据检查恢复结构与定量响应；单独保存与基线的差异及异常位置。
3. 计时区分每对重新准备参考和静态参考复用；把前端融合成本计入匹配阶段。
4. 候选通过相应检查后再改变默认；失败记录与代码一起保留。

默认入口目前继续保留原亚像素规则。搜索边界上的病态二次拟合已定位到一个 WXST 异常点；新的拟合保护属于算法规则变更，应单独立项测试，不混入这次封装。
