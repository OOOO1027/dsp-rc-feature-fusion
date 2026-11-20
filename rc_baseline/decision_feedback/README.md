# Phase 6：判决反馈双分支均衡 + NLTCP 格形解码

结果与结论见 [Phase 6 报告](../runs/phase6_decision_feedback/REPORT.md)，冻结协议见 [PROTOCOL.md](../runs/phase6_decision_feedback/PROTOCOL.md)。

| 文件 | 内容 |
|---|---|
| `dfe.py` | 第一遍判决、反馈特征、交叉拟合、3 状态 Viterbi、评分 |
| `experiment.py` | `prepare / smoke / validate / freeze / test` 五个模式 |
| `test_dfe.py` | 6 项合成数据单元测试（不读真实数据） |

**运行环境说明**：这三个文件与 2026-11-19 在原工作区 `/Users/oliver/Desktop/DSP` 实际运行的版本逐字节相同，哈希记录在 `runs/phase6_decision_feedback/frozen.json`。为保持与冻结记录一致，未改写成本仓库的跨电脑加载方式：

- 原始数据哈希按 `audit_outputs/2026-10-16/file_index.csv` 中的绝对路径核对；
- index 517–535 的前端读取自 `~/Documents/ChatGPT/DSP/deliverables/DSP_Vpp九点与跨偏振_20261113/inputs/`，本仓库未包含；
- `frozen.json` 中 `rc_baseline/src/dsp_rc/*.py` 的哈希对应原工作区版本，本仓库的 `dsp_rc` 含跨电脑加载修改，因此 `test` 模式在仓库内会因哈希不一致拒绝运行。

单元测试可直接在仓库中运行：

```bash
cd rc_baseline/decision_feedback
python -m unittest test_dfe -v
```
