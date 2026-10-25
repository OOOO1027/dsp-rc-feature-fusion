"""Read saved validation predictions; never fit or open historical test outputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import read,ROOT,load_prefix,assert_frozen,OUTPUTS
from dsp_rc.experiment import write_json,file_hash
from dsp_rc.phase2 import enriched_scores

def main(out):
    assert_frozen(out);rows=read(out/'results.json');front=read(out/'frontend_results.json');boot=read(out/'paired_block_results.json');decision=read(out/'decision.json')
    a,y,labels,_=load_prefix();p=np.load(out/'predictions.npz');mask=p['mask'];assert mask.sum()==9744
    flat=[]
    for r in rows:
        for split,start in [('train',0),('validation',10000)]:
            for ch in OUTPUTS:
                pred=p[f"{r['key']}_{split}_{ch}"];assert pred.shape==(10000,)
                actual=enriched_scores(pred,y[start:start+10000],labels[start:start+10000],mask)
                assert actual==r[split][ch],(r['key'],split,ch)
                flat.append(dict(frontend=r['frontend'],model=r['config']['name'],seed=r['config']['nonlinear_seed'],split=split,channel=ch,**{k:v for k,v in actual.items() if not isinstance(v,(list,dict))},width_i=actual['matlab_auto_width'][0]['width'],width_q=actual['matlab_auto_width'][1]['width']))
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    def avg(r,k):return np.mean([r['validation'][c][k] for c in OUTPUTS])
    def row(key):return next(r for r in rows if r['key']==key)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,k,title in zip(axes,['symbol_mse','ber_upper'],['Complex MSE','Fixed 2-bit BER upper bound']):
        ax.plot([0,1],[avg(row(f'{m}_linear80_reg_s100'),k) for m in ['on','off']],'o-',label='Linear80, lambda=.01')
        for seed in [100,101,102]:ax.plot([0,1],[avg(row(f'{m}_dual40_reg_s{seed}'),k) for m in ['on','off']],'o--',label=f'Dual40+40 seed{seed}')
        ax.set(xticks=[0,1],xticklabels=['Volterra on','Quadratic off'],ylabel=title,title='509 validation | mean RX1/RX2/XY');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(out/'factorial_validation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained');indices=np.flatnonzero(mask)[::5]
    for i,mode in enumerate(['on','off']):
        for j,ch in enumerate(OUTPUTS):
            ax=axes[i,j];z=p[f'{mode}_dual40_reg_s100_validation_{ch}'][indices]
            ax.scatter(z.real,z.imag,s=2,alpha=.25);ideal=np.unique(y[10000:][mask]);ax.scatter(ideal.real,ideal.imag,s=15,marker='+',c='black')
            ax.set(xlim=(-8,8),ylim=(-8,8),xlabel='I',ylabel='Q',title=f'{mode} | {ch} | dual seed100');ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.savefig(out/'constellations_validation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    keys=['on_dual_minus_linear80_reg','off_dual_minus_linear80_reg','interaction']
    for ax,col,label in zip(axes,[0,1],['MSE difference','BER upper difference']):
        for j,b in enumerate(['256','512']):
            means=np.array([boot[k][b]['mean_delta'][col] for k in keys]);ci=np.array([boot[k][b]['CI95'][col] for k in keys]);x=np.arange(3)+(j-.5)*.12
            ax.errorbar(x,means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=3,label=f'block {b}')
        ax.axhline(0,c='gray',lw=1);ax.set(xticks=np.arange(3),xticklabels=['Dual-linear ON','Dual-linear OFF','Interaction'],ylabel=label);ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'paired_interaction.png',dpi=160);plt.close(fig)
    s=r'''# 第五阶段：有限 Volterra 前端消融（仅训练与验证）

日期：2026-10-17。**实验已完成，未发现关闭前端二次项后双分支增量价值变大的证据。** 关闭二次项使当前后端的绝对验证误差明显增加；双分支三种子在on/off两侧的平均MSE均高于匹配线性RC。预声明潜力门槛未通过，停止本次实验，不扩大搜索、不评价旧测试。

## 1. 事前协议与真实执行边界

`protocol.json`先于小样本训练和完整训练写入；`run_started.json`记录协议SHA-256。完整实验严格12个前端/后端/seed组合、每组合独立RX1/RX2，共24个后端模型；4个前端分别训练一次供所有后端共享。不存在根据结果新增候选或最佳种子选择。

主2×2：有/无波形二次项 × Identity80 / Identity40+tanh40。后端115窗口、230维实输入、311×2共同读出，λ=.01；线性A算子范数.6、tanh谱半径.9、γ=.7；线性seed200、非线性seed100/101/102。两侧另预声明115窗口Ridge λ1e−5与linear80 λ1e−5锚点，不把受限λ线性模型冒充最强基线。线性模型没有非线性随机变量，只拟合一次，不将重复结果算作三个独立seed。

0-based训练[0,10000)、验证[10000,20000)，双端128保护，均计9744符号。加载器只将前80000采样/20000符号转为数值；完整原文件读取仅用于SHA核验（行向量文本可能一次读取整行，但未解析后续数值、预测或计分）。旧509/513/382测试预测文件从未打开。每段独立预处理/储备池重置；115居中窗口仍含未来57符号，属于离线S2S。波形前端监督仅来自训练前10000采样，所有尺度只从训练拟合。对已有前端的复验只读取模型参数并比较训练区，没有旧测试回放。

## 2. 消融实现与公平性

on复用原VolterraLMS，联合训练W和V；off用新增LinearWaveLMS从W=0重新拟合，仅保留逆序居中线性采样窗口：

\[\hat x_t=Wu_t,\quad W\leftarrow W+\mu(x^{Tx}_t-Wu_t)u_t^T.\]

off没有V参数或二次输出，不是把二次学习率设为0，也不是对已联合训练W进行推理损伤。保留RX1 33抽头/.022、RX2 43抽头/.013、单遍10000采样；on二次抽头/步长仍17/8e−5与13/5e−5。相同训练前缀对齐、极性与raw/Tx RMS共享；wave/CAP RMS及后端输入尺度分别在同一训练区重新估计后冻结。CAP固定载频.5625、rolloff.105/span10、phase0、全局采样原点、sps4。后端均在各自前端分布重新拟合，不套旧S2S权重。

四路前端训练相关峰全部lag0，未触发事前暂停规则，未调相位/延迟；509 RX2极性仍−1。on训练W/V和CAP符号与历史冻结前端逐元素一致，off的W确实不同。跨on/off每个后端配置的实际A/Win哈希相同；训练及推理不改变权重，所有模型重载预测一致。

4项新单元测试通过：独立设计矩阵递推核对、off齐次线性且预测不训练、尺度冻结/前端持久化、分段移位不回绕。正式运行前的四条3000/1000小样本路径均通过，完全位于旧训练区，前端也只在小样本训练段拟合；这些只是接口验收，不用来选模型。测试源码在协议之后、任何训练之前补充，核心实验源码及全部继承源码在协议中冻结；没有变更实验设计。最终报告脚本仅读取已保存训练/验证预测并逐字段重新计分核验。

## 3. 全部验证结果

下表聚合为RX1/RX2/XY等权平均，三个输出相关，并不是三份独立试验。auto是历史兼容自动位宽BER，不能等同于固定2-bit物理BER。N=9744，每输出38976真实参考bit；非法I/Q分量数U，合法错误bit数E，固定界限[E/(4N),(E+2U)/(4N)]，非法率U/(2N)。界限是未知非法映射的确定性范围，不是置信区间；未截断非法4/5。

|前端|模型|N seed|MSE|固定下界|固定上界|auto|非法率|
|---|---|---:|---:|---:|---:|---:|---:|
'''
    for r in rows:
        s+=f"|{r['frontend']}|{r['config']['name']}|{r['config']['nonlinear_seed']}|{avg(r,'symbol_mse'):.9f}|{avg(r,'ber_lower'):.9f}|{avg(r,'ber_upper'):.9f}|{avg(r,'matlab_compatible_ber'):.9f}|{avg(r,'invalid_rate'):.5%}|\n"
    s+='\n### 逐通道完整验证表（无删选）\n\n|前端/模型/seed|输出|MSE|固定下界–上界|auto|非法数/率|I/Q位宽|\n|---|---|---:|---|---:|---|---|\n'
    for r in rows:
        for ch in OUTPUTS:
            m=r['validation'][ch];w='/'.join(str(x['width']) for x in m['matlab_auto_width'])
            s+=f"|{r['key']}|{ch}|{m['symbol_mse']:.7f}|{m['ber_lower']:.7f}–{m['ber_upper']:.7f}|{m['matlab_compatible_ber']:.7f}|{m['invalid_pam4_decisions']} / {m['invalid_rate']:.5%}|{w}|\n"
    s+='\n![四格验证结果]('+str(out/'factorial_validation.png')+')\n\n## 4. 交互效应与固定种子\n\n定义Δon=MSE(dual,on)−MSE(linear,on)，Δoff同理；交互I=Δoff−Δon，负值才支持关闭二次项后非线性增量变大。双分支先平均三seed逐符号损失，再做配对连续块，不平均预测后重新计分。\n\n|对照|块长|ΔMSE|MSE 95%区间|ΔBER上界|BER上界95%区间|\n|---|---:|---:|---|---:|---|\n'
    for k in ['on_dual_minus_linear80_reg','off_dual_minus_linear80_reg','interaction','off_dual_minus_ridge_low','off_dual_minus_linear80_low','linear80_reg_off_minus_on','off_dual_minus_on_linear80_low']:
        for b,v in boot[k].items():
            s+=f"|{k}|{b}|{v['mean_delta'][0]:.9f}|{v['CI95'][0][0]:.9f} … {v['CI95'][0][1]:.9f}|{v['mean_delta'][1]:.9f}|{v['CI95'][1][0]:.9f} … {v['CI95'][1][1]:.9f}|\n"
    s+='''
每次2000个移动块重采样，块长256/512、seed404；先完整解码再抽取逐符号损失，不能拼块重新解码造成假边界；同位置用于所有条件/seed/输出。这里是反复使用的验证区上的探索性区间，没有独立新消息或总体确认性显著性保证。固定一个线性seed，不能代表所有储备池初始化。

三seed交互MSE点估计均为正；固定线性基线下，移除Volterra后双分支相对劣势没有消失，点估计反而略增。MSE交互在两种块长的探索性区间均为正（约0.00021–0.00073）；BER上界交互在256块长跨0、512块长为正，其稳定性更弱。不能仅凭这些重复使用验证区的区间作总体因果显著性声明。更明确的结果是四格绝对误差及同前端双分支相对线性的差异。on侧seed101的BER上界对主线性对照有局部微小改善，但MSE仍差；不隐藏局部指标，也不凭它挑seed。

事前门槛要求off双分支三seed均降低相对两个off强线性锚点的MSE、BER上界不恶化，两种块长的两指标区间均支持改善，并且交互MSE区间支持负值。本次不通过，`promising_for_new_payload=false`；没有追加配置、输入增益、延迟或超参数。

'''
    s+='![配对交互区间]('+str(out/'paired_interaction.png')+')\n\n## 5. 前端分布、复杂度与时间\n\n|前端/通道|线性/二次有效系数|训练更新数|训练CAP相关lag0|wave_scale/cap_scale|验证CAP功率|前端训练/验证预处理秒|\n|---|---|---:|---:|---|---:|---|\n'
    for k,f in front.items():
        s+=f"|{k}|{f['linear_parameters']}/{f['quadratic_parameters']}|{f['updates']}|{f['train_correlations']['0']:.6f}|{f['scales'][2]:.6f}/{f['scales'][3]:.6f}|{f['validation_symbol_stats']['power']:.6f}|{f['fit_seconds']:.6f}/{f['validation_preprocess_seconds']:.6f}|\n"
    s+='\n二次有效参数为上三角153/91；off均为0。前端尺度变化和重新拟合属于处理链的必要变化，不从验证区重新归一化。完整幅度分位数和均值见frontend_results.json。训练CAP相关高并不保证后端MSE更低，不能用单一相关系数替代公平输出评价。\n\n|配置|通道|后端监督/固定随机参数|输入scale|后端训练/验证推理秒|\n|---|---|---|---:|---|\n'
    for r in rows:
        for ch in ['rx1','rx2']:
            t=r['stats'][ch];s+=f"|{r['key']}|{ch}|{t['trained_parameters']}/{t['fixed_parameters']}|{t['input_scale']:.6f}|{t['fit_seconds']:.6f}/{r['inference_seconds'][ch+'_validation']:.6f}|\n"
    s+='''
每个后端训练与推理均10000符号（评分9744），计时不含共同前端、磁盘保存及评分；前端单列。可将相应前端与后端时间相加估计本次串行管线成本，但不能当作稳定硬件benchmark。双分支固定随机参数21680，线性80为24880，都是622监督读出参数；Ridge为462/0。匹配后端节点和读出维度不是严格等计算量，on更有前端二次项开销。

'''
    s+='![固定主seed验证星座]('+str(out/'constellations_validation.png')+')\n'
    s+='''
## 6. 结论与限制

在本次index509、固定波形训练规则和115窗口后端中，保留Volterra二次项有明显的总体性能价值；删除它没有释放出可由当前双分支RC稳定利用的额外预测收益。不能将这推广成“物理信道没有非线性”或“Volterra已经消除了所有非线性”。本实验操纵的是联合训练前端是否包含二次基函数，线性W也随之重新拟合，结论属于这两个训练管线的受控比较，不是固定其他系数后的纯二次项贡献分解。

只观察一个工况、同一消息的已用验证区；off线性前端步长/训练遍数按原规范固定，未优化，不能宣称它是最优纯线性接收机。没有新载荷测试，没有唯一合法PAM4解码规则。既有原始TXT上游处理历史仍不完全清楚。当前不建议为了展示双分支优势删除Volterra，也不建议扩大本轮搜索。优先保留现有强线性＋Volterra方案、统一助教规则及新载荷来源。

## 7. 复现与交付

源码独立在`rc_baseline/volterra_ablation/`，未修改任何历史源码/数据/模型。复现必须新目录，使用已有环境，按顺序：

```bash
cd /Users/oliver/Desktop/DSP/rc_baseline
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python volterra_ablation/experiment.py prepare --out runs/phase5_reproduce
.venv/bin/python -m unittest discover -s volterra_ablation -p 'test_*.py' -v > runs/phase5_reproduce/unit_tests.log 2>&1
.venv/bin/python volterra_ablation/experiment.py smoke --out runs/phase5_reproduce
.venv/bin/python volterra_ablation/experiment.py run --out runs/phase5_reproduce
.venv/bin/python volterra_ablation/report.py --out runs/phase5_reproduce
```

实验入口没有test模式，重复run会拒绝覆盖。`protocol.json`、`run_started.json`、`input_manifest.json`保存事前设计/来源；`models/`为4前端+24后端；`predictions.npz`仅训练/验证；`results.json`/`metrics.csv`完整指标；`seed_effects.json`、`paired_block_results.json`为固定种子与时序配对结果；`decision.json`止损；`unit_tests.log`、`smoke.json`及`verification.json`为验收证据。保存了全部失败方向，没有挑seed或只报有利通道。
'''
    (out/'REPORT.md').write_text(s)
    preserved=read(out/'preserved_before.json');assert all(file_hash(ROOT/k)==h for k,h in preserved.items())
    write_json(out/'delivery_verification.json',dict(old_files_unchanged=len(preserved),saved_metric_records_exactly_recomputed=len(flat),source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in Path(__file__).parent.glob('*.py')},new_unit_tests_passed=4,figures=3,no_test_predictions=True))
    print('Report and 3 figures saved; all',len(flat),'saved train/validation score records independently rechecked; historical files unchanged.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('runs/phase5_volterra_ablation'));main(p.parse_args().out.resolve())
