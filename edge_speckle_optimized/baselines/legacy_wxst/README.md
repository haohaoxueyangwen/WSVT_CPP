# 冻结的 WXST GPU 原型

三个原型Python文件保留原字节。v1分阶段前端，v2前端融合，v3再增加候选缓存；原型缺透射率。`legacy_complete.py` 是此前比较用的独立完整输出适配器，不修改v3。

完整适配器只调整库定位到本目录，并记录该来源转换；旧实验目录中的适配器没有改动。

复现需要与Orin兼容的PyTorch、NumPy、SciPy，以及本机库：

```bash
# 从工程包根目录执行；历史目标不在默认构建中。
cmake --build build --target wxst_match wxst_optimized wxst_cached -j2
```

库写入本目录，文件名与原型期望一致。原型命令仍保留自己的旧参数与输出协议，不自动冒充edge统一CLI；阅读源文件或用原实验脚本复现历史批次。与edge比较时使用相同数学配置及完整输出适配器。
