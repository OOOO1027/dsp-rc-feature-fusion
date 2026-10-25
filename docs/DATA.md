# 数据、来源和完整性

本仓库携带全部已报告实验工况的复现依赖，而非原始采集全集。

- `rxdata/`：NLTCP index 509、513、382，各有 RX1/RX2，一共 6 个原始文本。
- 每路 204800 实采样；RF Tx 参考也为 204800 实采样。
- `v1.txt`、`v2.txt` 与两份 `decimal_PAM4_*.txt` 各 51200 项；sps=4。
- 三个工况共用上述 Tx/符号/标签。不同 index 不构成不同消息。
- `txdata/` 原目录完整保留，包括额外 Tx 波形及 WFM。不同哈希不证明不同载荷；缺少第二套完整配对标签，不能作为新盲测。
- 保留原 MATLAB `.m` 与采集表（若在原数据树中存在），不含无关课程作业、试卷、讲义或其他组 PPT。
- 原 572 对中的其余 569 对 RX 不打包，其他制式的接收数据不打包。用户原本的 RAR、ZIP 和完整采集目录不改动。

`audit_outputs/2026-10-16/file_index.csv` 是原始完整审计，保留原绝对路径及哈希。仓库新增 `dsp_rc.portable.load_audit_index` 只把原项目根映射为当前 checkout 根；不改数据内容，不跳过 SHA-256 校验，不按同名文件猜测。

`archive/import_manifest.json` 记录所有导入文件的原始字节哈希。`archive/portability_changes.json` 逐项登记交付副本的路径适配与文档链接改动。原源码与顶层原文档另存为 `archive/original_sources_and_docs.zip`。数据、模型、预测及历史 runs 文件维持导入哈希。

`MANIFEST.sha256.json` 是本次交付文件清单。`python tools/verify_bundle.py` 检查每个文件、历史导入一致性以及所有打包原始数据与审计哈希的一致性。不把索引里未打包的采集误报为丢失。

更换新数据必须新增索引和 payload 证据，不能用新文件覆盖旧路径。非法判决/边界说明见顶层 TA_QUESTIONS.md。
