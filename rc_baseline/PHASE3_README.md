# 第三阶段：双分支 RC 特征融合

完整实现及实测结果：[实验报告](/Users/oliver/Desktop/DSP/rc_baseline/runs/phase3_feature_fusion/REPORT.md)。

入口：`src/dsp_rc/fusion.py`（模型）、`src/dsp_rc/phase3.py`（分阶段协议）、`src/dsp_rc/phase3_report.py`（只读结果制图）。保留第一、二阶段的 README、源码及结果。

选定主模型：线性40节点＋tanh40节点，115符号输入窗口，共同读出311×2，λ=0.01，线性seed200、tanh seed100。每个物理通道独立拟合，共享超参数；物理RX1/RX2最后固定等权合并。

结果不支持稳定优于强线性基线。index509仅作旧保留集回归；513/382按元数据预选、算法冻结后评价，但共用Tx载荷，不是独立新消息盲测。

复现时先进入本目录，使用现有`.venv`：

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m dsp_rc.phase3 prepare --out runs/phase3_reproduce
.venv/bin/python -m dsp_rc.phase3 smoke --out runs/phase3_reproduce
.venv/bin/python -m dsp_rc.phase3 search --out runs/phase3_reproduce
.venv/bin/python -m dsp_rc.phase3 fit-extra --out runs/phase3_reproduce
.venv/bin/python -m dsp_rc.phase3_numerics --out runs/phase3_reproduce
.venv/bin/python -m dsp_rc.phase3 evaluate --out runs/phase3_reproduce
.venv/bin/python -m dsp_rc.phase3_report --out runs/phase3_reproduce
```

模型接口（z为用对应冻结前端预处理后的完整单段复数CAP符号）：

```python
from dsp_rc.fusion import FusionModel
model = FusionModel.load('runs/phase3_feature_fusion/index509/dual40_reg_rx1.npz')
result = model.transform(z)
# result['input']: (T, 230)
# result['states']['linear'], ['nonlinear']: 各 (T, 40)
# result['features']: (T, 311) = [h_linear, h_nonlinear, input, bias]
# result['prediction']: (T,) complex，原理想符号尺度
# result['final_states']: 两分支末状态；正式每个split重置，不跨分段沿用
```

评价继续使用原NLTCP解码；非法4/5不强制映射。必须同时报告自动位宽BER、非法率、固定2-bit上下界。完整模型、配置、源文件哈希、预测及时间都在实验目录；`metrics.csv`包含所有候选与最终统计。
