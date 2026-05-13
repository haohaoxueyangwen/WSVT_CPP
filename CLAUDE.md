# WSVT_CPP

X 射线散斑追踪相位恢复 C++ 实现，从 Python 版本 (WXSVT_v2) 翻译而来。

## 项目概述

WSVT_CPP 实现了基于小波变换的 X 射线散斑追踪（Wavelet-based X-ray Speckle Vector Tracking）算法，用于从同步辐射 X 射线成像数据中提取相位、位移场、微分相位衬度（DPC）、透射率和暗场信号。

### 核心算法

- **WXST** — 单帧小波 X 射线散斑追踪求解器（`wsvt::WXST` 类）
- **WSVT** — 多帧小波 X 射线散斑追踪求解器，堆叠多帧后联合求解（`wsvt::WSVT` 类）

### 处理流水线

```
输入图像 → 裁剪/对齐 → 金字塔降采样 → Template Window 堆叠 → 小波变换 → 相关匹配求位移 → Frankot-Chellappa 相位恢复
```

每一步的 C++ 对应：

| 步骤 | Python | C++ 模块 |
|------|--------|----------|
| 图像加载 | `load_image()` (PIL) | `io_image.cpp` / `io_h5.cpp` |
| 裁剪 ROI | `image_roi()` | `crop_ops.cpp` |
| 图像对齐 | `phase_cross_correlation` + `fourier_shift` | `align_ops.cpp` |
| 金字塔降采样 | `pywt.dwtn('db3')` | `pyramid.cpp`（支持 `Mean2x2` 和 `Db3Aa` 模式） |
| Template Window | `stack_TemplateWindow()` | `template_window.cpp` |
| 小波变换 | `Wavelet_transform_multiprocess()` (pywt) | `wavelet_ops.cpp`（支持 db2/db3/db6，含 HWD 原生接口） |
| 相关匹配 | `dist_numba()` + `find_disp()` | `euclidean_dist.cpp` + `core.cpp`（`find_disp`） |
| 相位恢复 | `frankotchellappa()` | `phase_recovery.cpp`（FFTW3 加速或暴力 DFT 后备） |
| 位移上采样 | `resampling_spline()` (RectBivariateSpline) | `solver_utils.cpp`（双三次插值） |
| 滤波去噪 | `filter_erosion()` | `filter_ops.cpp` |

### 输出数据

`SolverOutput` / `WXSTOutput` 结构体包含：

| 字段 | 说明 |
|------|------|
| `displace_x/y` | 位移场（像素） |
| `DPC_x/y` | 微分相位衬度 |
| `phase` | 恢复相位（rad） |
| `transmission` | 透射率图像 |
| `darkfield` / `darkfield_nd` | 暗场信号 |
| `time_cost_s` 等 | 各阶段计时 |

## 构建

```bash
cd WSVT_CPP
cmake --preset release
cmake --build --preset release
```

### 构建预设

| 预设 | 用途 |
|------|------|
| `debug` | 调试构建 |
| `release` | 发布构建 |
| `release-lto` | 发布 + 链接时优化 |
| `asan` | Address/UBSan 检测 |
| `tsan` | Thread Sanitizer 检测 |

### 依赖

| 依赖 | 必需？ | CMake 选项 | 说明 |
|------|--------|------------|------|
| OpenMP | 必需 | 自动检测 | 多线程并行 |
| HDF5 | 可选 | `WSVT_WITH_HDF5` | H5 文件读写 |
| OpenCV | 可选 | `WSVT_WITH_OPENCV` | 图像读写和斜率估计（`use_estimate=true`） |
| FFTW3 (single) | 可选 | `WSVT_WITH_FFTW` | 快速 Frankot-Chellappa 相位恢复；缺失时回退到 O(n⁴) 暴力 DFT |

Conda 环境下自动检测 `$CONDA_PREFIX/lib`。

## 运行

### CLI 命令

```bash
# 内置 demo（合成数据）
wsvt_cli demo

# 从图像目录运行 WSVT（多帧）
wsvt_cli wsvt_dir <sample_dir> <ref_dir> <out_dir> [--key value ...]

# 从图像目录运行 WXST（单帧）
wsvt_cli wxst_dir <img_dir> <ref_dir> <out_dir> [--key value ...]

# 从 HDF5 文件运行
wsvt_cli wsvt <img_h5> <img_key> <ref_h5> <ref_key> <out_dir> [--key value ...]
wsvt_cli wxst <img_h5> <img_key> <ref_h5> <ref_key> <out_dir> [--key value ...]
```

### Python 启动器

`launch.py` 支持配置文件驱动运行和结果导出：

```bash
python launch.py wsvt_dir --config config.json
python launch.py wsvt_dir <img_dir> <ref_dir> <out_dir> --set n_cores=16 --set crop=512
```

运行后自动将 HDF5 结果导出为 PNG/TIFF。

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `crop` / `m_image` | 512 | 裁剪/图像尺寸 |
| `n_s` | 5 | 尺度数（WXST 专用） |
| `cal_half_window` | 20 | 相关窗口半宽 |
| `n_s_extend` | 4 | 尺度扩展 |
| `n_cores` | 4 | OpenMP 并行线程数 |
| `n_group` | 4 | 分组大小（控制内存） |
| `energy` | 14000 | X 射线能量 (eV) |
| `p_x` | 0.65e-6 | 像素尺寸 (m) |
| `z` | 0.5 | 传播距离 (m) |
| `wavelet_level_cut` | 2 | 小波层级截断 |
| `pyramid_level` | 2 | 金字塔层级 |
| `n_iter` | 1 | 迭代次数 |
| `save_img` | false | 保存 TIFF 输出图像 |
| `use_wavelet` | true | 启用小波处理 |
| `use_estimate` | false | 启用斜率估计（需 OpenCV） |
| `cleansave` | false | 仅保存位移（WSVT 模式） |

## 项目结构

```
WSVT_CPP/
├── include/wsvt/           # 公共头文件（22 个）
│   ├── core.hpp             # find_disp 子像素峰值定位
│   ├── common.hpp           # 索引辅助、clamp、双三次插值、HWD 张量操作
│   ├── types.hpp            # Shape2D/3D、Layout 枚举、FFTW RAII 封装
│   ├── image.hpp            # Image2D/Tensor3D（ owning）+ ImageView2D/TensorView3D（非 owning view）
│   ├── aligned_alloc.hpp    # 64 字节对齐分配器 AlignedVector
│   ├── export.hpp           # WSVT_API 动态库导出宏
│   ├── wavelet_ops.hpp      # 小波变换（db2/db3/db6，含 HWD 原生接口）
│   ├── pyramid.hpp          # 金字塔降采样 + PyramidResult
│   ├── phase_recovery.hpp   # Frankot-Chellappa 相位恢复
│   ├── euclidean_dist.hpp   # 负欧氏平方距离相关
│   ├── template_window.hpp  # Template Window 堆叠
│   ├── solver_utils.hpp     # 斜率估计（OpenCV）、stddev/PV 统计
│   ├── crop_ops.hpp         # 裁剪操作
│   ├── align_ops.hpp        # 图像对齐
│   ├── filter_ops.hpp       # 滤波去噪
│   ├── io_h5.hpp            # HDF5 读写
│   ├── io_image.hpp         # TIFF 图像读写
│   ├── io_json.hpp          # JSON 参数读写
│   ├── console_ops.hpp      # 终端彩色输出
│   ├── wsvt_pipeline.hpp    # WSVT 多帧流水线 + SolverOutput
│   └── wxst_pipeline.hpp    # WXST 单帧流水线 + WXSTOutput
├── src/                    # 源文件（18 个）
│   ├── main.cpp             # CLI 入口（demo/wxst/wsvt/wxst_dir/wsvt_dir）
│   ├── wsvt_pipeline.cpp    # WSVT 流水线实现
│   ├── wxst_pipeline.cpp    # WXST 流水线实现
│   ├── wavelet_ops.cpp      # 小波变换实现
│   ├── phase_recovery.cpp   # 相位恢复实现
│   ├── pyramid.cpp          # 金字塔降采样实现
│   ├── core.cpp             # find_disp 实现
│   ├── euclidean_dist.cpp   # 相关距离实现
│   ├── template_window.cpp  # Template Window 实现
│   ├── image_ops.cpp        # 图像操作
│   ├── filter_ops.cpp       # 滤波实现
│   ├── io_h5.cpp            # HDF5 读写
│   ├── io_image.cpp         # 图像文件读写
│   ├── io_json.cpp          # JSON 读写
│   ├── align_ops.cpp        # 图像对齐
│   ├── crop_ops.cpp         # 裁剪操作
│   ├── console_ops.cpp      # 终端输出
│   └── solver_utils.cpp     # 求解器辅助函数
├── tests/                  # Catch2 单元测试
│   ├── test_phase_recovery.cpp
│   ├── test_pyramid.cpp
│   ├── test_wavelet.cpp
│   ├── test_wavelet_hwd.cpp
│   └── test_wxst_smoke.cpp
├── bench/                  # nanobench 性能基准测试
│   └── bench_core.cpp
├── cmake/                  # CMake 配置
│   └── wsvtConfig.cmake.in
├── launch.py               # Python 启动脚本（配置文件 + 结果导出）
├── run.sh                  # Bash 启动脚本
├── config.json             # 运行配置示例
├── vcpkg.json              # vcpkg 依赖清单（hdf5/opencv 特性）
└── CMakePresets.json       # CMake 预设
```

## 核心数据结构

### 图像/张量（`image.hpp`）

- `Image2D<T>` — 拥有数据的 2D 图像（row-major）
- `ImageView2D<T>` — 非拥有 2D 视图（零拷贝参数传递）
- `Tensor3D<T, Layout>` — 3D 张量，支持 `CHW` 和 `HWD` 布局
- `TensorView3D<T, Layout>` — 3D 张量视图
- `Shape2D` / `Shape3D` — 维度描述

### 内存布局约定

- **CHW**：通道优先，用于金字塔数据（`[ch, h, w]`）
- **HWD**：深度在后，用于小波系数数据（`[h, w, depth]`），每个像素的深度维度连续

### 对齐分配（`aligned_alloc.hpp`）

`AlignedVector<T, 64>` — 64 字节对齐的 `std::vector` 替代，用于 SIMD 友好的热路径数据。

## 测试

```bash
cmake --preset debug -DWSVT_BUILD_TESTS=ON
cmake --build --preset debug
ctest --preset debug
```

### 使用 testdata 测试

```bash
# 多帧 WSVT（100 帧 2048x2048 16-bit TIFF）
wsvt_cli wsvt_dir testdata/sample testdata/ref testdata/output_wsvt \
    --save_img true --n_cores 16 --crop 512

# 单帧 WXST
wsvt_cli wxst_dir testdata/sample_single testdata/ref_single testdata/output_wxst \
    --save_img true --n_cores 16
```

输出保存为 HDF5 文件（`WSVT_result.hdf5` / `WXST_result.hdf5`）和可选 TIFF 图像。

## 编码规范

- C++20 标准，无扩展
- 编译警告：`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`
- Release 额外优化：`-O3 -march=native -funroll-loops -ffinite-math-only`
- 动态库符号隐藏，通过 `WSVT_API` 宏显式导出
- 所有公共 API 在 `namespace wsvt` 下
