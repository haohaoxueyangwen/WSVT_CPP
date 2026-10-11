# 原 FASTWSVT-cpp / FASTWXST-cpp CPU 对照

`cpp_original_snapshot.tgz` 是此前从主服务器提取的53个原始文件，哈希见 `cpp_source_manifest.json`；`cpu_driver.cpp` 是独立C接口，不修改算法源码。`CMakeLists.txt` 为当时同机对照的外部构建项目。二进制不随源码包分发。

原CPU要求C++20、OpenMP和单精度FFTW/线程库。Orin系统g++9不足，既有隔离GCC12工具链位于原 `cpu_compare_20261011/toolchain`；GPU工程本身使用系统g++9/C++17。原CPU53文件保持原服务器字节，不用Windows换行版本覆盖。

同机已构建库和配置位于 `/home/xray/Documents/WSVT_GPU_orin_20261010/cpu_compare_20261011/build/libcpu_driver.so` 和该目录 `build/CMakeCache.txt`。运行已构建比较器时，既有环境需要隔离工具链的libstdc++预加载：

```bash
cd /home/xray/Documents/WSVT_GPU_orin_20261010/cpu_compare_20261011
LD_PRELOAD="$PWD/toolchain/lib/libstdc++.so.6" /usr/bin/python3 compare.py --help
```

本目录保存可审查的CPU基线来源；不声称已将原CPU所有CLI参数接入GPU统一入口。复制源码后构建应复用已验证的CMake配置、FFTW路径和禁用HDF5选项，并另记构建哈希。CPU包装器当前固定历史几何与算法配置，只用于原比较协议。
