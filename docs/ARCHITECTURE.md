# 模型与工程架构

每个物理通道的链路：接收波形 → 训练限定同步/极性/尺度 → 波形线性 LMS＋二次 Volterra → CAP/RRC/下采样 → 115 符号窗口 → 两储备池 → 特征拼接 → 共同 Ridge → I/Q 复符号 → NLTCP 解码。

双分支为 `Identity40 + tanh40`。输入 230 实维，拼接 `[L40,H40,u230,1]`，输出矩阵 311×2、正则 0.01。线性矩阵算子范数 0.6、leak 0.7，递归收缩界 0.72；非线性谱半径 0.9。矩阵固定，只拟合读出。物理 RX1/RX2 各自拟合，最后 0.5/0.5 合并。

|模块|职责|
|---|---|
|`core.py`|同步、CAP、波形/符号 LMS、单储备池、NLTCP 与 BER|
|`experiment.py`|原数据加载与校验、冻结 ChannelPipeline、Phase1|
|`portable.py`|仓库版新增，仅迁移审计路径，不变更算法|
|`phase2.py`|单分支诊断、有限搜索、历史确认|
|`fusion.py`|LinearReservoir、FusionModel、共同输出层、保存加载|
|`phase3.py`|候选、冻结、多工况与历史测试入口|
|`residual_fusion.py`|FeatureBank、ResidualFusion、投影与分组正则|
|`phase4*.py`|验证残差研究、固定种子、统计诊断与报告|
|`volterra_ablation/`|仅移除二次项并重拟合线性前端的 Phase5|

接口：`FusionModel.fit(z,y,mask)` 仅训练；`extract(z)` 导出特征；`transform(z)` 返回特征及预测；`predict(z)` 不接受标签；`save/load` 保存实际矩阵与尺度。输入须是同一冻结前端生成的完整独立分段，不跨 split 传递末状态。

残差方案冻结线性基础，预测训练残差，输出 `y_linear + eta*r_hat`；验证四家族选 eta=0。不能把诊断用非零 eta 模型标记成赢家。

本次三处代码改动仅替换审计字典的路径构造。原历史协议记录的是当时源码哈希，不应修改这些协议来适配新源码。重新运行时必须先在新目录 prepare，生成当前源码对应的新协议；不要直接在旧 runs 下执行 verify/evaluate/report 写回归档。旧脚本内保留原绝对路径的报告模板、MATLAB 路径属于历史工具，不是跨平台认证。Python 新入口见 REPRODUCIBILITY.md。
