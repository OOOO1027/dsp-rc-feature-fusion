# 第四阶段：残差驱动融合研究

[完整中文报告](/Users/oliver/Desktop/DSP/rc_baseline/runs/phase4_residual_fusion/REPORT.md)

结论：预定义的四组残差方案均选择eta=0；固定三种子检查未支持非线性残差预测的稳定增益。停止扩大搜索，保留强线性基线。本轮没有重新预测或评价旧测试集。

新增模块：

- `residual_fusion.py`：两个固定储备池、残差修正、训练统计、近似正交化、分组/等价惩罚、模型持久化。
- `phase4.py`：仅训练/验证的分阶段实验（24主要候选、eta零基线、固定种子、配对移动块）。
- `phase4_diagnostics.py`：残差结构与Tx来源盘点。
- `phase4_verify.py`：旧文件哈希、旧验证预测重放、数值条件检查。
- `phase4_report.py`：读取保存结果生成图表和报告。

运行目录为本目录，沿用`.venv`，完整复现命令见报告。新输出目录必须不存在。所有原有源码与前三阶段结果保持不变。

接口示例：

```python
from dsp_rc.residual_fusion import ResidualFusion
m = ResidualFusion.load('runs/phase4_residual_fusion/models/res_linear80_low_raw_eta0.25_rx1_s100.npz')
# z必须是对应冻结前端产生的完整独立分段CAP复符号；不传入标签。
base, correction = m.components(z)
pred_diagnostic = m.predict(z)       # 已保存的eta=0.25，仅为负面诊断候选
pred_linear = m.predict(z, eta=0)    # 本轮四组验证选择实际支持“不修正”
```

另有两份本地清单（未对外发送）：

- [新载荷数据要求](/Users/oliver/Desktop/DSP/rc_baseline/runs/phase4_residual_fusion/NEW_PAYLOAD_REQUEST.md)
- [NLTCP非法输出与计错规则待确认](/Users/oliver/Desktop/DSP/rc_baseline/runs/phase4_residual_fusion/NLTCP_DECODER_QUESTIONS.md)
