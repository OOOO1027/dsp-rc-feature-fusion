# index=509 单分支 RC 基线

本目录是独立 Python 实现，不修改上级 `Data/` 和 `NN数据20250320/`。完成了训练、验证、独立测试；采用 **S2S** 单分支 tanh RC，不含双分支。

完整结果与方法说明：[第一阶段实验报告](PHASE1_REPORT.md)。

## 复现

当前环境已在本目录 `.venv` 配置 Python 3.12、NumPy、SciPy、Matplotlib。准确版本在 `requirements.lock.txt`。不依赖 TensorFlow、GPU 或 MATLAB。

在 `/Users/oliver/Desktop/DSP/rc_baseline` 运行：

```bash
# 数值机制单元检查
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v

# 仅使用正式训练区前5000符号的小样本流程；输出目录必须尚不存在
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m dsp_rc.experiment --smoke --out runs/smoke_reproduce

# 冻结配置的完整复现，不是新一轮调参
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m dsp_rc.experiment --out runs/index509_reproduce
```

新机器先用 Python 3.11+ 创建独立 venv，再安装锁定依赖：`python -m pip install -r requirements.lock.txt`。然后使用该 venv 的 Python 执行以上命令。`--root` 指定 DSP 目录（须包含数据和原审计文件索引），`--config` 可指定配置 JSON。程序会核对7份输入的 SHA-256；不匹配则中止。原始数据没有复制进本目录。

现有 `runs/index509_fixed_v1` 是已经看过测试结果的基线记录，后续可固定配置复验，但不要在其测试区上反复选参数。后续调参只能用训练/验证区；最终发表比较应增加新采集或新的封存测试集。

## 代码结构

- `src/dsp_rc/core.py`：同步/极性、RRC/CAP、LMS/Volterra、符号 LMS、RC 状态、岭读出、NLTCP 判决、BER。
- `src/dsp_rc/experiment.py`：配置检查、原文件哈希、训练专用 fit、持久化、冻结推理、统一评分、图表。
- `configs/index509.json`：本轮在测试前固定的唯一正式参数配置。
- `tests/test_core.py`：10项数值机制/边界/标签隔离检查。
- `runs/smoke_509`：小样本验证，全部位于正式训练区。
- `runs/index509_fixed_v1`：正式模型、运行日志、预测、指标、图表。

输出目录必须新建，程序拒绝覆盖已有实验。模型是无 pickle 的 NumPy NPZ；预测文件不复制原始波形。

## 单分支机制和特征融合接口

`Reservoir.transform(x, initial_state=None)` 返回：

|返回值|形状|含义|
|---|---|---|
|states|T×80|固定 A、Win 下的 tanh 储备池状态|
|features|T×131|`[states(80), input(50), bias(1)]`|
|final_state|80|末时刻状态；当前实验每个split重置为0|

`ChannelPipeline.fit(rx_train, tx_train, y_train)` 只接收训练区数据。`ChannelPipeline.transform(raw, start_symbol)` 不接收标签，返回：

- `cap_symbols`: T 个复数，共同前端生成，使用训练区缩放；
- `input`: T×50，25符号居中窗口，按 `[I(t-12),Q(t-12),...,I(t+12),Q(t+12)]` 排列；
- `states`、`features`、`final_state`；
- `before`、`lms`、`rc`: T 个复数预测，I/Q 使用原理想六电平尺度。

恢复模型并导出某一通道/分段的接口示例：

```python
from pathlib import Path
import json
import numpy as np
from dsp_rc.experiment import ChannelPipeline

base = Path('/Users/oliver/Desktop/DSP/rc_baseline')
run = base / 'runs/index509_fixed_v1'
cfg = json.loads((run / 'config.json').read_text())
model = ChannelPipeline.load(run / 'model_rx1.npz', 'rx1', cfg)
# raw_segment 必须是同一分段的4*T个实数；start_symbol为全局0-based符号起点。
# 原文件的来源与SHA-256见 input_manifest.json；不要给transform传入标签。
features = model.transform(raw_segment, start_symbol=10000)
states = features['states']
```

下一阶段在每个物理通道内部，分别运行线性激活储备池和 tanh 储备池，再将两条状态按相同 symbol IDs 拼接。输出层仍拟合 I/Q，而非 PAM4 整数。必须沿用相同 split、mask、前端和缩放；共同输入/偏置只拼接一次。线性储备池的稳定性、两分支节点预算等尚未定义，也未实现。RX1/RX2 合并仍是另一层操作，不等于算法内部双分支。

## 评价语义

原解码公式可能产生非法 PAM4 值4/5，本实现原样保留。`matlab_compatible_ber` 逐分量按最大整数自动决定比特位宽，再平均两分量 BER；本轮完整测试均为3 bit，不应误称为每 PAM4 符号固定2 bit的物理 BER。

`ber_lower` / `ber_upper` 固定参考分母为4×有效复符号数：合法判决按自然二进制精确计错；每个非法 PAM4 判决暂记0～2个未知比特错误，给出确定性界限。它们不是置信区间，也不是经修复后的解码结果。只有 `invalid_pam4_decisions == 0` 时 `ber_exact` 才给出唯一固定2 bit BER。没有擅自选择裁剪、取模或另一种有约束解码器。课程最终采用哪一种非法判决映射，需要与助教统一；现有结果足以比较本轮各方法，不能隐去该差异。
