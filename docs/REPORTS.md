# 阶段报告索引

这些是只调整本地绝对图片/文件链接的阅读副本，指标和结论保持原文。历史原件保存在 rc_baseline/runs。

- [phase1](reports/phase1.md)
- [phase2_509](reports/phase2_509.md)
- [phase3_feature_fusion](reports/phase3_feature_fusion.md)
- [phase4_residual_fusion](reports/phase4_residual_fusion.md)
- [phase5_volterra_ablation](reports/phase5_volterra_ablation.md)

原文中的绝对运行目录是历史环境记录，执行命令请以 [跨电脑复现指南](REPRODUCIBILITY.md) 为准。

## 验证集误差互补分析（2026-10-30）

- [报告、图表及复现方式](../rc_baseline/runs/validation_complementarity_20261030/REPORT.md)
- [分析前协议](../rc_baseline/runs/validation_complementarity_20261030/protocol.json)
- [核验记录](../rc_baseline/runs/validation_complementarity_20261030/verification.json)

复用 index=509 的既有验证预测，无重新训练、无测试指标计算。两分支有少量互相救错，但尚无稳定可识别的分工；现有证据不足以支持立即启动长记忆线性＋短时非线性训练。报告保留初次读取单行标签文件时触及测试后缀文本、随后改为验证范围流式读取的完整说明。

## Phase 6：判决反馈双分支 + Viterbi（2026-11-19）

- [报告：结果、测试表、结论边界](../rc_baseline/runs/phase6_decision_feedback/REPORT.md)
- [冻结协议](../rc_baseline/runs/phase6_decision_feedback/PROTOCOL.md)、[偏离记录](../rc_baseline/runs/phase6_decision_feedback/deviations.md)
- [代码与运行说明](../rc_baseline/decision_feedback/README.md)

## Phase 7 / 7b：前端全量训练 × 软反馈 × 局部非线性分支（2026-11-20）

- [Phase 7b 报告：27 个从未评分过的新工况上的确认结果](../rc_baseline/runs/phase7b_guarded_frontend/REPORT.md)
- [Phase 7b 冻结协议](../rc_baseline/runs/phase7b_guarded_frontend/PROTOCOL.md)
- [Phase 7 停止记录](../rc_baseline/runs/phase7_frontend_softfeedback/STATUS.md)、[382 前端发散诊断](../rc_baseline/runs/phase7_frontend_softfeedback/ANOMALY_382.md)
- [代码与运行说明](../rc_baseline/phase7b/README.md)
