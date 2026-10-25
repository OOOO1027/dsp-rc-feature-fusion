# 第二组交接：线性 RC＋非线性 RC 特征融合

交接编号：DSP-G2-20261016-01。状态：材料已核对，供团队接手；未创建新任务、未对外发送。项目根目录 `/Users/oliver/Desktop/DSP`，本目录不是Git仓库。交接范围为现有代码、结果、复现入口和待确认问题；不授权覆盖历史结果或继续参数搜索。

## 1. 可以直接向团队说明的结论

第二组已完成指定结构的实现和受控验证：在每个接收通道内建立Identity线性RC与tanh非线性RC，拼接状态，由一个共同Ridge输出层恢复理想复符号。115窗口、共同前端、共同分段/保护区/解码下，与窗口Ridge、单线性RC、单tanh RC及固定参数LMS比较。

**双分支优于本次提供参数下的LMS，但没有稳定超越强线性Ridge或线性RC。** 后续残差预测、近似正交化、标准化和分组惩罚也未建立稳定非线性增量收益；四个残差家族都选择eta=0，已止损。负面结果有完整配置、预测、模型和日志支持，属于研究成果，不应换一张局部有利的图宣称获胜。

## 2. 本组到底做了什么

共同前端是波形线性LMS＋二次Volterra，再CAP解调得到复符号。双分支工作在这一前端之后，是S2S，不是原64APSK例程的W2S。每个时刻取115复符号居中窗口，交错I/Q为230维实输入，按训练最大绝对值缩放。

两分支都递推状态：`h=(1-gamma)*h_prev+gamma*activation(A@h_prev+Win@[u,1])`。线性用Identity并将A算子范数限制为0.6，非线性用tanh且谱半径0.9，gamma均0.7。线性有效递归范数上界0.72，避免直接替换激活造成发散。A/Win固定，隐藏层不反向传播。

主模型 `dual40_reg` 的两路各40节点，特征为 `[L40,H40,u230,1]`，共同读出311×2，λ=.01，种子200/100。共同Ridge使两组特征在同一个监督目标下分配线性系数，可保留互补信息、抑制冗余；这并不保证一定优于单分支。两个分支分别输出后加权属于另一种方法，本组主要交付是特征融合，没有引入MLP。

每个物理RX独立拟合尺度和输出权重，再固定0.5/0.5合并预测为XY；不能把这两路物理接收通道称作Identity/tanh双分支。

![已有结构图](/Users/oliver/Desktop/DSP/rc_baseline/runs/phase3_feature_fusion/architecture.png)

## 3. 已做消融及解释边界

|研究|实现/对照|结果|
|---|---|---|
|强线性基线|115窗口Ridge；线性RC80/160|已接近或优于非线性方案，必须保留|
|非线性单RC|tanh80/160|115窗口修复旧RX2上下文不足，未建立稳定非线性增量|
|直接特征拼接|40+40、80+80；同总节点/读出维度、同λ对照|不稳定胜过最强线性；预算匹配不等于等计算量|
|输入直通|保留一份输入/偏置 vs 仅状态|纯状态80+80验证上界约.3804，明显失败|
|残差修正|Ridge或线性RC基础冻结，非线性预测训练残差，eta含0|四家族均选0；最优非零设计3种子MSE均略差|
|状态残差化|训练拟合H对输入/L的Ridge投影，验证冻结使用|确实去掉线性可解释部分，但验证残差增量R²仍负|
|标准化/分组Ridge|训练统计；输入/L/H惩罚.01/1/10|数值条件改善，预测未胜强线性|
|等价坐标控制|同时变换特征和惩罚|预测差约1.18e−11，证明单纯坐标变化不是新能力|

原因应按证据强弱表达：长窗口已提供大量可线性利用的信息；线性状态约99.9%方差能由当前窗口解释；非线性状态虽有约18–19%不同变化，未能在验证区预测线性残差。额外参数在训练区可拟合残差却未泛化。共同前端已含Volterra可能是机制之一，**目前没有做前端消融，不能写成已查明的因果原因**。

本组最有价值的技术结论是：把“比来源LMS好”“状态不重复”“数值更稳定”“有非线性增量”四件事分开验证；建立强线性、匹配节点、坐标等价、eta=0和固定种子的控制，避免把参数增多或口径变化当结构贡献。

## 4. 交接文件导航

以下路径均相对于 `rc_baseline/`，保留原目录结构。

|入口|用途|
|---|---|
|`src/dsp_rc/core.py`、`experiment.py`|数据加载、训练限定同步/前端、CAP、窗口、NLTCP与评分|
|`src/dsp_rc/fusion.py`|LinearReservoir、FusionModel，第二组基础实现|
|`src/dsp_rc/phase3.py`|预声明、18候选、冻结、额外工况、统一历史测试|
|`src/dsp_rc/residual_fusion.py`|FeatureBank、ResidualFusion、近似投影/坐标变换/分组惩罚|
|`src/dsp_rc/phase4.py`|仅训练/验证的24配置、eta零控制、固定seed及止损|
|`src/dsp_rc/phase4_diagnostics.py`、`phase4_verify.py`|残差/载荷盘点、预测一致性和保护检查|
|`tests/`|历史累计23项测试；第四阶段最终日志23/23|
|`runs/phase3_feature_fusion/`|protocol、validation_results、frozen_algorithm、frozen_all_models、3个index目录、完整metrics.csv|
|`runs/phase4_residual_fusion/`|protocol、candidate_results、selection、seed_results、decision、paired_block_bootstrap、metrics.csv、56个models及训练/验证预测|

模型API：`FusionModel.fit(z,y,mask)`仅训练；`extract(z)`返回input、states、features、final_states；`transform(z)`增加prediction；`predict(z)`无标签入口；`save/load`保存实际矩阵与尺度。主模型状态各(T,40)、特征(T,311)、输出(T,) complex。`ResidualFusion.components(z)`返回基础预测与单位eta修正，`predict(z,eta=0)`回到基础。每次传入**完整独立分段**，不要切成无上下文小块，也不要跨split传递末状态。

## 5. 五分钟只读接手核验（不训练）

先读本文件和[研究总结](/Users/oliver/Desktop/DSP/PROJECT_RESEARCH_SUMMARY.md)，打开Phase3/4报告及图表。下面命令只读保存的结果，不调用fit、不重新评价测试：

```bash
cd /Users/oliver/Desktop/DSP/rc_baseline
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('runs/phase4_residual_fusion')
r=json.loads((p/'candidate_results.json').read_text())
s=json.loads((p/'selection.json').read_text())
d=json.loads((p/'decision.json').read_text())
assert len(r)==24 and not d['test_evaluated']
assert not any(v['pass_gate'] for v in d['methods'].values())
print(s['residual_family_winners_including_eta0'])
print(json.loads((p/'verification.json').read_text()))
PY
```

预期四组赢家为对应 `ridge_low` 或 `linear80_low`，无非零残差赢家。23项通过是保存日志的历史证据，不是这条只读命令重新运行了23项测试。

## 6. 队友需要重新运行时的完整入口

**下列命令会训练；本次交接没有执行。** 必须使用新的、尚不存在的输出目录。不能用旧输出目录覆盖，也不能把旧数据重跑称为新盲测。现有 `.venv` 可用，锁定依赖见 `requirements.lock.txt`（NumPy2.5.4、SciPy1.18.1、Matplotlib3.11.2）；不需要TensorFlow。跨机器不能复制虚拟环境本身，须在副本建立兼容Python环境。

```bash
cd /Users/oliver/Desktop/DSP/rc_baseline
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# 第二组基础方法：先核对输出目录不存在
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m dsp_rc.phase3 prepare --out runs/phase3_team_reproduce
.venv/bin/python -m dsp_rc.phase3 smoke --out runs/phase3_team_reproduce
.venv/bin/python -m dsp_rc.phase3 search --out runs/phase3_team_reproduce
.venv/bin/python -m dsp_rc.phase3 fit-extra --out runs/phase3_team_reproduce
.venv/bin/python -m dsp_rc.phase3_numerics --out runs/phase3_team_reproduce
# 仅历史测试重放，不用于选模
.venv/bin/python -m dsp_rc.phase3 evaluate --out runs/phase3_team_reproduce
.venv/bin/python -m dsp_rc.phase3_report --out runs/phase3_team_reproduce

# 残差研究：没有test模式
.venv/bin/python -m dsp_rc.phase4 prepare --out runs/phase4_team_reproduce
.venv/bin/python -m unittest discover -s tests -v > runs/phase4_team_reproduce/unit_tests_final.log 2>&1
.venv/bin/python -m dsp_rc.phase4_diagnostics --out runs/phase4_team_reproduce
.venv/bin/python -m dsp_rc.phase4 smoke --out runs/phase4_team_reproduce
.venv/bin/python -m dsp_rc.phase4 search --out runs/phase4_team_reproduce
.venv/bin/python -m dsp_rc.phase4 seeds --out runs/phase4_team_reproduce
.venv/bin/python -m dsp_rc.phase4_verify --out runs/phase4_team_reproduce
.venv/bin/python -m dsp_rc.phase4_report --out runs/phase4_team_reproduce
```

Phase4依赖已有Phase1冻结前端和Phase3参考结果；不能只拷贝`residual_fusion.py`。完整包需保留 `src/tests/configs`、历史runs、审计目录、匹配的原始数据。初始复现与Phase2命令见 `README.md`、`runs/phase2_509/REPORT.md`，无需为查看本组结果全部重跑。

**路径限制：** `load_assets`以审计CSV中的绝对路径索引并校验SHA-256；默认根目录从源码位置推导。换到队友电脑后，必须在工作副本建立旧路径→新路径映射，逐文件保持原hash，并让副本加载器/副本索引使用该映射；不能只改文件名或跳过校验，也不能改原审计归档。本地Markdown绝对图片链接迁移后也需在文档副本替换根路径。当前没有自动跨机器迁移工具，不能声称下载后任意目录即跑。

验收依次看：文件hash/长度和符号ID正确；split/mask相同；保存重载预测一致；历史参考预测匹配（不同BLAS可能仅浮点容差）；候选数和eta选择一致；所有BER分母/位宽/非法数齐全；新输出无覆盖。计时无需逐秒一致。

## 7. 团队展示及后续边界

可以展示：既有架构图、窗口诊断、三个条件星座/BER界限、完整消融、残差图、eta含零图、数值一致性和止损决策。推荐讲述“实现→公平基线→局部失败→残差证伪”，不把主任务成功偷换成必须获得正提升。

最终报告仍需助教确认评价契约及课程目标；若要主张新消息泛化，需要不同Tx载荷的完整匹配包。当前可先完成带限制的课程研究报告，不必为了交差强行重启搜索。Volterra前端消融仅有静态方案，详见[NEXT_EXPERIMENT_FEASIBILITY.md](/Users/oliver/Desktop/DSP/NEXT_EXPERIMENT_FEASIBILITY.md)。

接手开场白可复制：`$project-handoff 恢复。项目 /Users/oliver/Desktop/DSP；来源 GROUP2_HANDOFF.md，编号 DSP-G2-20261016-01。先只读核对四份交接文档与 Phase3/4 保存结果，确认统计口径和缺项；不训练、不扩大搜索、不改历史成果。返回差异及需要助教确认的事项。`

本材料采用[project-handoff skill](/Users/oliver/.codex/skills/project-handoff/SKILL.md)组织；只保存交接资料，未发送消息或启动接手任务。
