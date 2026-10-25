# 有限 Volterra 消融

已完成，仅使用 index509 的训练和验证区。结果见 [REPORT.md](/Users/oliver/Desktop/DSP/rc_baseline/runs/phase5_volterra_ablation/REPORT.md)。

`experiment.py`：prepare先冻结协议，smoke验证接口，run执行固定四格和事前锚点；没有test模式。
`test_ablation.py`：4项off线性性/递推、尺度冻结、持久化和分段测试。
`report.py`：只读取保存预测，复核指标、生成表格与图。

off保留且从零重新训练波形线性LMS，没有二次项；on/off每侧独立训练尺度及后端读出。保留历史文件，不扩大搜索。主2×2和固定三seed未支持双分支增量优势，维持停止结论。完整复现命令在报告，必须使用新的输出目录。
