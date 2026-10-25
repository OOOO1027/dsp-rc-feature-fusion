# 复现层级与运行说明

## 1. 查看结果：无需训练

阅读顶层总结和 docs/REPORTS.md。`rc_baseline/runs/` 含五阶段配置、实际模型矩阵、预测、日志、完整消融和结果，直接查阅即可。

根目录执行：

```bash
python tools/verify_bundle.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/verify_bundle.py --replay-validation
```

前者只核验文件哈希；后者额外使用冻结前端和双分支模型，对 509 验证段重放预测，比较历史保存数组（不重新计算 BER，不训练，不预测测试段）。不同 BLAS 允许 1e-10 数值容差。仅此检查通过不等于已完整重跑所有实验。

## 2. 单元测试：仅合成数据

安装环境后，在仓库根目录：

```bash
python -m unittest discover -s rc_baseline/tests -v
python -m unittest discover -s rc_baseline/volterra_ablation -p 'test_*.py' -v
```

这些测试可能拟合小型合成数组，用来核验代码，不是对真实工况重新训练或搜索。

## 3. 重新训练复现：按需人工启动

下面是归档研究的复现入口，不是建议恢复搜索。本次打包没有执行这些命令。先安装项目，然后：

```bash
cd rc_baseline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# 以下所有 team_* 输出目录在首次运行前必须不存在。

# Phase1（完整运行包含历史测试评价）
python -m dsp_rc.experiment --out runs/team_phase1

# Phase2：复用归档 Phase1 冻结前端
python -m dsp_rc.phase2 search --out runs/team_phase2
# 仅明确需要重放历史测试时执行，不能用于选模：
# python -m dsp_rc.phase2 confirm --out runs/team_phase2

# Phase3：第二组基础方法，先冻结协议
python -m dsp_rc.phase3 prepare --out runs/team_phase3
python -m dsp_rc.phase3 smoke --out runs/team_phase3
python -m dsp_rc.phase3 search --out runs/team_phase3
python -m dsp_rc.phase3 fit-extra --out runs/team_phase3
python -m dsp_rc.phase3_numerics --out runs/team_phase3
# 仅用于历史测试重放；执行前明确它不是新盲测：
# python -m dsp_rc.phase3 evaluate --out runs/team_phase3
# python -m dsp_rc.phase3_report --out runs/team_phase3

# Phase4：仅训练/验证，继承仓库中归档的 Phase1/3 参照
python -m dsp_rc.phase4 prepare --out runs/team_phase4
python -m unittest discover -s tests -v > runs/team_phase4/unit_tests_final.log 2>&1
python -m dsp_rc.phase4_diagnostics --out runs/team_phase4
python -m dsp_rc.phase4 smoke --out runs/team_phase4
python -m dsp_rc.phase4 search --out runs/team_phase4
python -m dsp_rc.phase4 seeds --out runs/team_phase4
python -m dsp_rc.phase4_verify --out runs/team_phase4
python -m dsp_rc.phase4_report --out runs/team_phase4

# Phase5：独立入口，没有 test 模式
python volterra_ablation/experiment.py prepare --out runs/team_phase5
python -m unittest discover -s volterra_ablation -p 'test_*.py' -v > runs/team_phase5/unit_tests.log 2>&1
python volterra_ablation/experiment.py smoke --out runs/team_phase5
python volterra_ablation/experiment.py run --out runs/team_phase5
python volterra_ablation/report.py --out runs/team_phase5
```

每段应独立执行并检查退出状态，失败后不要继续粘贴下一段。Phase3 报告依赖 evaluate 输出，所以默认不执行；Phase4/5 报告仅处理训练/验证。

这些命令各自复现一个阶段，后续阶段按原代码依赖**归档基线**，不是自动改用上面新生成的 team_phase1/3。若要更换依赖，须明确修改实验协议并另立结果目录，不能把不同前端或参照混用。

## 4. 复现限制

- 包含实际训练参数及数据，不保证不同平台浮点结果逐 bit 一致。环境记录在各 runs 的 environment JSON。
- 历史协议、原代码快照和哈希保留；仓库适配版三处加载器哈希不同是已登记变更，不修改旧哈希来制造一致性。
- MATLAB 共享矩阵验证曾通过，CAP/biterr 工具箱完整对照未完成。MATLAB 文件保留原运行路径，跨机器需在副本调整路径及工具箱，不列为 Python 快速开始的依赖。
- 若要主张新消息泛化，需要新的完整 Tx/RX/标签包。重跑旧数据不能得到新的盲测证据。
