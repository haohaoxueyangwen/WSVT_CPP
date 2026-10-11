# 工程交接：2026-10-11

用户已确认所需数据区域可用，要求从软件工程角度整理几个版本。操作范围为新目录 `D:/Yangrc/code/WSVT_project/edge_speckle`，Orin镜像 `/home/xray/Documents/WSVT_GPU_orin_20261010/edge_speckle`。

原根仓库 `codex/wg-umpa-research-20260903` / `31a247e3f00ea32f4e840a2e237b06b551cf5841`，嵌套 `codex/cpp-wsvt` / `c6768f47fac21970437409ab9ac4339bf65052f7`；两处预存脏修改均保留，无提交、重置或删除。根项目3份主文档不覆盖，本包同名文档记录增量。

已整理：包内原生依赖、CMake构建/构建哈希、独立WSVT/WXST配置与入口、参数拒绝策略、输出原点/单位、失败状态、计时范围、历史版本与CPU来源快照、原始JSON证据索引。原生算法仅变更include路径；runtime变更库路径、CPU样条初始化失败时的句柄清理保护及几何来源说明。

验证命令（Orin，从包根执行）：

```bash
/usr/bin/python3 tools/build.py --jobs 2
/usr/bin/python3 -m unittest discover -s tests -v
/usr/bin/python3 -m tests.smoke_orin --original-edge ../edge_adaptation_20261011 \
  --wxst-input ../bb11_pair256.npz \
  --wsvt-input ../edge_adaptation_20261011/wsvt100_fixture128.npz \
  --u16-input ../edge_adaptation_20261011/cli_input_u16_128.npz \
  --out tests/results/NEW_CHECK
/usr/bin/python3 tools/check_sources.py --original-root ..
```

四组包装回归通过：七字段相同、缓冲复用、输出独立、生命周期和线程拥有规则。已有CPU/GPU异常点不改写；WSVT重排默认关闭；未新跑全尺寸基准。源码包不包含数据/大输出，实际ARM库只在Orin本机构建。

下一项：添加长期会话采集示例，明确参考图更新与样品队列策略。任何新参数/亚像素规则变更另建候选，先保留原结果和协议。
