# 偏离协议记录

## D1（2026-11-19）：冻结后修复测试汇总代码中的 KeyError

- **现象**：第一次运行 `test` 时，在 index509 测试段完成预测与评分后、构造汇总行时抛出 `KeyError: 'symbols'`（`selected_metrics` 字典未包含符号数）。进程在写出 `test_results.json` 或打印任何指标之前退出；`test.log` 只含 traceback。没有任何测试指标被显示、保存或用于决策。
- **修改**：`experiment.py` 中一行，`selm['xy']['symbols']` 改为 `metrics['ridge_low']['xy']['symbols']`（两者使用同一 mask，数值相同）。只影响汇总行中"错 bit 数"字段；模型、特征、候选、选择结果、评分函数均未改动。
- **处理**：原 `frozen.json` 改名为 `frozen_superseded_reporting_bug.json` 保留；重新运行 `freeze` 生成新的 `frozen.json`（选中配置不变，仅 `experiment.py` 哈希变化）；随后重新运行一次 `test`。
- **影响评估**：测试段被重新计算一次，但第一次运行没有产生可见结果，不存在根据测试结果调整方法的可能。

## 澄清（非偏离）

- H3 的参照"同来源最佳 DF-tap"按"crossfit 来源 + past 范围"解释，因为 DF-RC 只使用过去判决；实现见 `validate()`。
