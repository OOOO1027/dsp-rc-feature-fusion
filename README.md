# DSP 第二组：线性／非线性 RC 特征融合

NLTCP 光通信课程项目的可复现研究归档与团队协作仓库。**方法二：同一接收通道内，Identity 与 tanh 储备池提取状态，拼接后使用共同 Ridge 输出层。** 不是两个物理通道的加权，也不是两个分支分别预测后加权。

**结论：双分支优于本次来源参数下的符号 LMS，但没有稳定超过 115 窗口强线性 Ridge／线性 RC。残差四家族选择 eta=0；删除 Volterra 二次项使性能退化，也未释放稳定双分支增量。保留 Volterra，停止扩大当前搜索。**

本仓库只打包已有研究并修复跨目录加载；没有重新训练真实数据模型。当前未完成 LMS 系统调参、X/Y alpha 遍历，也未实现方法一的输出加权或方法三的 MLP 融合。

## 先看哪些文件

|入口|内容|
|---|---|
|[研究总结](PROJECT_RESEARCH_SUMMARY.md)|五阶段结论、统一对照、数学定义及证据边界|
|[第二组交接](GROUP2_HANDOFF.md)|结构、接口和历史工作范围；安装以本 README 为准|
|[下一步决策](NEXT_EXPERIMENT_FEASIBILITY.md)|为什么保留 Volterra、停止搜索|
|[助教问题](TA_QUESTIONS.md)|非法 PAM4、新载荷与课程评价目标|
|[复现指南](docs/REPRODUCIBILITY.md)|只读验收与重新运行的区别、分阶段命令|
|[数据说明](docs/DATA.md)|包含的数据、缺失范围、原始哈希|
|[架构说明](docs/ARCHITECTURE.md)|算法、模块与团队协作约定|
|[阶段报告索引](docs/REPORTS.md)|可在 GitHub 查看图片的报告副本|
|[PPT](presentations/DSP_Group2_五阶段研究汇报_最终版.pptx)|12 页汇报|
|[讲稿与问答](presentations/DSP_Group2_逐页口头汇报稿与问答.md)|约 10 分钟逐页讲稿、20 组问答|

## 快速开始（默认不训练）

```bash
git clone https://github.com/OOOO1027/dsp-rc-feature-fusion.git
cd dsp-rc-feature-fusion
# 仅标准库即可核验仓库内容；不训练、不推理、不计分
python3 tools/verify_bundle.py

# Python >=3.11；虚拟环境不上传、不直接复制其他电脑的环境
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./rc_baseline
# 使用冻结模型重放 validation 预测，与归档预测比较；不训练、不评价测试
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/verify_bundle.py --replay-validation
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`；线程环境变量按 PowerShell 语法设置。原研究依赖版本保存在 `rc_baseline/requirements.lock.txt`。精确环境复刻可先 `python -m pip install -r rc_baseline/requirements.lock.txt`，再安装项目；若目标 Python／平台没有对应 wheel，不应声称环境完全相同。普通安装使用项目依赖，数值相等需通过上面的重放检查，不能由安装成功推断。

本次交付在原有 macOS Python 环境验证，未做 Windows／Linux 实机认证。源码是目录可迁移的；历史 JSON 内原路径仅作为溯源字段保留。

## 目录架构

```text
rc_baseline/
  src/dsp_rc/           # 数据链、单 RC、双分支、残差模型及各阶段入口
  configs/             # 固定 index509 配置
  tests/               # 合成数据单元测试
  volterra_ablation/    # Phase5 独立实现、测试和说明
  matlab/              # MATLAB/Python 共享矩阵核对脚本
  runs/                # 原始五阶段模型、预测、指标、日志、协议和图表
NN数据20250320/NN数据20250320/
  rxdata/              # 仅实验使用的 509、513、382，各 RX1/RX2
  txdata/              # 原 Tx 参考目录及载荷盘点材料
  *.txt / *.m          # 理想 I/Q、两份 PAM4 标签、原 MATLAB 函数
Data/                  # 原始 MATLAB RC 参考代码（其他调制制式，不当作 NLTCP 真值）
audit_outputs/         # 原始审计索引、元数据、哈希和覆盖清单
archive/               # 原源码/文档快照、导入清单和迁移变更记录
document_history/      # 第五阶段整理前的历史文档
presentations/         # 已交付 PPT 和讲稿
docs/                  # 跨电脑复现说明及 GitHub 可读报告镜像
tools/                 # 无训练的数据完整性和验证预测重放工具
```

数据和结果直接随 Git 下载，无 Git LFS、无外部网盘步骤。只纳入复现现有研究所需的三个采集工况，**不包含原 572 对接收数据全集**。完整原始数据仍保留在原工作区；审计索引中其他路径代表曾经审计过，不代表此仓库含该文件。详见数据说明。

## 结果与评价口径

index509 **历史测试 auto BER**，不能当成固定 2-bit 物理 BER：

|模型|RX1|RX2|XY 等权|
|---|---:|---:|---:|
|来源参数 LMS|0.043789|0.027561|0.013605|
|窗口 Ridge|0.018404|0.022153|0.006156|
|线性 RC80|0.018410|0.022078|0.006097|
|非线性 RC80|0.018550|0.022525|0.006366|
|双分支 40+40|0.018523|0.022595|0.006194|

- 训练 1–10000，验证 10001–20000，历史测试 20001–51200；双端保护 128，每段独立预处理和状态重置。
- 非法 PAM4 输出 4/5 保持原样。N 个复符号、合法错误 bit 数 E、非法分量数 U：固定 bit 界限 `[E/(4N), (E+2U)/(4N)]`，非法率 `U/(2N)`。该界限不是置信区间。
- 509/513/382 共用 Tx，各工况各自拟合训练段；不是独立新消息盲测，也不是零样本迁移。
- Phase4/5 仅验证研究。禁止把重跑旧测试解释成新盲测，不据旧测试继续选模。

## 团队协作

先看保存结果再决定是否运行。归档 `runs/` 不可覆盖；新实验输出到新的 `rc_baseline/runs/team_*` 目录（默认 Git 忽略）。共享新结果前添加协议、配置、seed、数据索引、评价口径和结论，使用分支与 PR，避免覆盖其他成员工作。未经新的明确研究计划，不自动启动扩大搜索。

仓库为私有，仓库所有者可在 Settings → Collaborators 邀请成员。原始课程数据及助教代码不自动授予公开再分发许可；本仓库未擅自添加覆盖第三方材料的开源许可证。
