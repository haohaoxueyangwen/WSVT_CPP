# WSVT_CPP 依赖说明

## 必需依赖
- C++17 编译器
- CMake 3.20+

## 可选依赖
- HDF5（启用 `write_h5/read_h5`）
- OpenCV（启用 `use_estimate=true` 的 `slope_tracking` 初始化）

## CMake 开关
- `WSVT_WITH_HDF5=ON/OFF`
- `WSVT_WITH_OPENCV=ON/OFF`

如果可选依赖未找到，工程仍可编译，但对应功能在运行时会抛异常。

## vcpkg（推荐）
项目提供 `vcpkg.json`，可按功能安装：
- `--x-feature=hdf5`
- `--x-feature=opencv`

示例（Windows）：
```powershell
vcpkg install --x-manifest-root=WSVT_CPP --feature-flags=manifests --x-feature=hdf5 --x-feature=opencv
```

## 启动脚本
- Python 启动：`launch.py`
- Shell 启动：`launch.sh`
- 配置文件模板：`launch_config.example.json`

配置文件运行示例：
```powershell
python launch.py --config launch_config.example.json
```

导出 PNG 结果图：
```powershell
python launch.py --config launch_config.example.json --save-png
```

说明：
- `--save-png` 现在由 C++ 直接导出 `.tif` 结果图（不依赖 `h5py`）
- 仍会同时保存 HDF5 结果文件

输入模式：
- HDF5：`img_h5/ref_h5 + img_key/ref_key`
- 文件夹：`img_dir/ref_dir`（自动读取图片并临时转为 HDF5）

文件夹支持格式：
- `.tif .tiff .png .jpg .jpeg .bmp .webp`
