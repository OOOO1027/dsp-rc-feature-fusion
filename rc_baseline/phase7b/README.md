# Phase 7 / 7b：前端全量训练 × 软判决反馈 × 局部非线性分支

- 结果与结论：[Phase 7b 报告](../runs/phase7b_guarded_frontend/REPORT.md)；冻结协议：[PROTOCOL.md](../runs/phase7b_guarded_frontend/PROTOCOL.md)
- Phase 7（`../phase7/p7.py`）在开发阶段发现 382 的前端训练发散后**在确认前停止**，见 [STATUS.md](../runs/phase7_frontend_softfeedback/STATUS.md) 与 [ANOMALY_382.md](../runs/phase7_frontend_softfeedback/ANOMALY_382.md)；Phase 7b 加入只用训练误差的稳定性判据后完成。

| 文件 | 内容 |
|---|---|
| `../phase7/p7.py` | 软判决、局部多项式分支、前端重拟合、访问守卫、评分（7b 只导入复用） |
| `p7b.py` | 稳定性判据与 F40g 前端；`prepare / smoke / develop / freeze / confirm` |
| `test_p7b.py`、`../phase7/test_p7.py` | 单元测试（含 509/382 训练段的真实数据检查） |

**运行环境说明**：与 Phase 6 相同，这些文件与 2026-11-20 在原工作区 `/Users/oliver/Desktop/DSP` 实际运行的版本逐字节相同（哈希见 `runs/phase7b_guarded_frontend/frozen.json`），未改写成本仓库的跨电脑加载方式：原始数据按审计表绝对路径核对；确认集与保留集的采集文件（40/50/60/70/75 mA）不在本仓库中；仓库内的 `dsp_rc` 含跨电脑修改，`confirm` 会因代码哈希不符拒绝运行。结果文件可直接查看。
