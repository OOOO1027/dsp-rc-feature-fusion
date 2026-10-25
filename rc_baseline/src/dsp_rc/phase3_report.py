"""Read saved artifacts only; creates report/figures/CSV, never fits or predicts."""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .phase3 import read
from .experiment import write_json,file_hash

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])+'\n'
def fmt(m):return f"{m['matlab_compatible_ber']:.6f}"
def bounds(m):return f"{m['ber_lower']:.6f}–{m['ber_upper']:.6f}"

def run(root,out):
    algorithm=read(out/'frozen_algorithm.json');results=read(out/'test_results.json')['conditions'];validation=read(out/'validation_results.json')
    byname={r['config']['name']:r for r in validation};winners=algorithm['family_winners'];names=list(winners.values());short={names[0]:'Ridge',names[1]:'tanh RC',names[2]:'Linear RC',names[3]:'Dual RC','lms':'LMS','before':'Before'}
    allnames=algorithm['evaluated_names'];metricsrows=[]
    for r in validation:
        for split in ['train','validation']:
            for ch,m in r[split].items(): metricsrows.append(dict(index=509,split=split,channel=ch,method=r['config']['name'],**m))
    for idx,record in results.items():
        for ch,ms in record['metrics'].items():
            for name,m in ms.items(): metricsrows.append(dict(index=idx,split='test',channel=ch,method=name,**m))
    columns=['index','split','channel','method','symbols','reference_bits','matlab_compatible_ber','ber_lower','ber_upper','invalid_rate','invalid_pam4_decisions','symbol_mse','legal_bit_errors','bit_errors_lower','bit_errors_upper']
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns+['I_width','Q_width'],extrasaction='ignore');writer.writeheader()
        for r in metricsrows:writer.writerow(dict(r,I_width=r['matlab_auto_width'][0]['width'],Q_width=r['matlab_auto_width'][1]['width']))
    # Fixed selection order and plot sampling; no performance-based subset.
    for idx,record in results.items():
        a=np.load(out/f'index{idx}/test_predictions.npz');ids=np.flatnonzero(a['mask'])[::8]
        for ch in ['rx1','rx2','xy']:
            fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
            for ax,name in zip(axes.ravel(),['before','lms']+names):
                z=a[ch+'_'+name][ids];ax.scatter(z.real,z.imag,s=2,alpha=.25,rasterized=True)
                i,q=np.meshgrid([-5,-3,-1,1,3,5],[-5,-3,-1,1,3,5]);ax.scatter(i,q,marker='+',c='black',s=14)
                ax.set(xlim=(-8,8),ylim=(-8,8),xlabel='I',ylabel='Q',title=f"{short[name]} | auto {fmt(record['metrics'][ch][name])} | k="+'/'.join(str(v['width']) for v in record['metrics'][ch][name]['matlab_auto_width']));ax.set_aspect('equal');ax.title.set_fontsize(10);ax.grid(alpha=.2)
            fig.suptitle(f'Index {idx} | {ch.upper()} | frozen evaluation, shared mask and scales')
            fig.savefig(out/f'index{idx}/constellation_{ch}.png',dpi=140);plt.close(fig)
        a.close()
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,(idx,record) in zip(axes,results.items()):
        for j,ch in enumerate(['rx1','rx2','xy']):
            means=[];lo=[];hi=[]
            for name in names:
                m=record['metrics'][ch][name];mid=(m['ber_lower']+m['ber_upper'])/2;means.append(mid);lo.append(mid-m['ber_lower']);hi.append(m['ber_upper']-mid)
            ax.errorbar(np.arange(4)+(j-1)*.15,means,yerr=[lo,hi],fmt='o',capsize=3,label=ch.upper())
        ax.set(xticks=np.arange(4),xticklabels=[short[n] for n in names],ylabel='Fixed 2-bit BER bounds',title=f'Index {idx}',yscale='log');ax.grid(alpha=.25);ax.legend(loc='center right')
    fig.suptitle('Deterministic mapping bounds, NOT confidence intervals');fig.savefig(out/'ber_bounds.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    vals=[sum(r['validation'][ch]['ber_upper'] for ch in ['rx1','rx2','xy'])/3 for r in validation]
    colors=[{'A':'#5a809e','B':'#449b8b','C':'#aa8754','D':'#8667aa'}[r['config']['family']] for r in validation]
    ax.barh(np.arange(len(validation)),vals,color=colors);ax.set(yticks=np.arange(len(validation)),yticklabels=[r['config']['name'] for r in validation],xlabel='Mean upper BER (RX1, RX2, XY), log scale',xscale='log',title='All 18 predeclared configurations | 509 validation only');ax.invert_yaxis();ax.grid(axis='x',alpha=.25);fig.savefig(out/'validation_search.png',dpi=160);plt.close(fig)
    # Simple exportable architecture drawing.
    fig,ax=plt.subplots(figsize=(13,5));ax.set(xlim=(0,13),ylim=(0,5));ax.axis('off')
    boxes=[(.1,2,2,1,'Frozen waveform front\nCAP -> 115-symbol\nwindow: 230 real dims'),(3,3.2,2.3,1,'Identity reservoir\n40 states | seed 200\ncontractive recurrence'),(3,.8,2.3,1,'tanh reservoir\n40 states | seed 100'),(6,2,2,1,'Concatenate\n[linear, tanh, x, 1]\n311 dimensions'),(9,2,1.6,1,'Shared Ridge\n311 x 2\nlambda = 0.01'),(11.3,2,1.6,1,'Complex symbols\nNLTCP decode\nBER + invalid rate')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc='#e8eff7',ec='#385475'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for a,b in [((2.2,2.5),(2.9,3.7)),((2.2,2.5),(2.9,1.3)),((5.4,3.7),(5.9,2.5)),((5.4,1.3),(5.9,2.5)),((8.1,2.5),(8.9,2.5)),((10.7,2.5),(11.2,2.5))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#385475'))
    ax.annotate('one input bypass',xy=(6.5,3.1),xytext=(.5,4.6),arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90',color='#777'),fontsize=9)
    ax.text(6.5,.15,'One dual-branch model per RX channel; final RX1/RX2 combination remains fixed 0.5 / 0.5.',ha='center',fontsize=10)
    fig.savefig(out/'architecture.png',dpi=160,bbox_inches='tight');plt.close(fig)
    doc=['# DSP 第三阶段：线性＋非线性 RC 特征融合\n',
    '**已完成实现、18项测试、小样本验证、18个预定义配置的验证搜索，以及冻结后的3工况统一评价。结论：双分支在本轮受控实验中没有稳定超越115窗口线性 Ridge 或最强单分支；不应宣称结构性优势。**\n',
    '新增代码和结果独立保存，前两阶段源码与结果的哈希全部保持不变。509为历史保留集回归；513和382为按采集元数据预选的额外工况，算法冻结后才评价。它们共用载荷，不能作为独立随机消息的盲测证据。\n',
    '## 1. 实现与数学定义\n',
    '![双分支结构](architecture.png)\n',
    '每个RX物理通道内，使用相同的CAP符号与115复符号居中窗口。230实维输入按训练有效窗口的绝对最大值缩放。两储备池独立固定随机矩阵；线性seed=200，非线性seed=100；两路物理RX使用相同seed策略，但各自拟合归一化与读出。\n',
    r'$$u_t=\operatorname{vec}_{IQ}(z_{t-57:t+57})/s_{train},\quad \tilde u_t=[u_t;1].$$'+'\n',
    r'$$h^L_t=(1-\gamma_L)h^L_{t-1}+\gamma_L(A_Lh^L_{t-1}+W^L_{in}\tilde u_t),$$'+'\n',
    r'$$h^N_t=(1-\gamma_N)h^N_{t-1}+\gamma_N\tanh(A_Nh^N_{t-1}+W^N_{in}\tilde u_t).$$'+'\n',
    r'$$F_t=[h^L_t,h^N_t,u_t,1],\quad W_{out}=(F_{tr}^{T}F_{tr}+\lambda I)^{-1}F_{tr}^{T}Y_{tr},\quad \hat z_t=(F_tW_{out})_1+j(F_tW_{out})_2.$$'+'\n',
    '代码以`solve`求解而不显式求逆，偏置也正则化。主方法拼接状态后，保留原RC的一份输入直通与一份偏置；没有把两个分支各自完整features重复拼接。另单列严格状态拼接`[hL,hN,1]`消融。所有输出是原始I/Q尺度，再经现有NLTCP判决；测试不做增益重估。\n',
    '### 线性分支稳定性\n',
    r'令随机矩阵 $G$，设置 $A_L=\alpha G/\|G\|_2$，$0\leq\alpha<1$、$0<\gamma_L\leq1$。于是有效递归 $B_L=(1-\gamma_L)I+\gamma_L A_L$ 满足 $\|B_L\|_2\leq1-\gamma_L+\gamma_L\alpha<1$。这比只限制特征值更强，保证初态扰动收缩且有界输入不会发散。'+'\n',
    '线性配置中的`linear_norm`是算子范数目标，**不是谱半径**；真实谱半径同时落盘。选定40节点线性分支：α=0.6、γ=0.7，A谱半径0.306885，B谱半径0.500085、2-范数0.626822；保守收缩界0.72，128步初态残留界5.48e−19。它只界定初态差，不是对持续输入或全部边界伪影的误差保证。每段重置确保不跨split携带状态。tanh分支沿用A谱半径0.9、γ=0.7。\n',
    '## 2. 数据、同步和评估独立性\n',
    '原始路径由既有`load_assets`和审计CSV确认，每个条件7文件校验SHA-256；204800波形采样/4=51200复符号。复用共同波形LMS/Volterra、CAP/RRC、固定phase=0；本实验仍是S2S，并非W2S。前端本身已有二次Volterra项，因此所谓“线性基线”仅指符号均衡器，不代表整条接收机全线性。\n',
    table(['index','bias / VPP','用途','元数据位置'],[['509','30 mA / 200 mV','训练＋验证开发；旧测试回归','等高线图 row8'],['513','30 mA / 300 mV','预选幅度变化工况','等高线图 row9'],['382','50 mA / 200 mV','预选偏置变化工况','等高线图 row28']]),
    '三行AWG/OSC均10/10 G、roll-off=0.105、衰减13/15/2。选择时只读取条件列1–8和index列12，排除历史BER列9–11；完整对文件由审计索引确认。协议时间和内容先于模型搜索与新工况波形加载保存。\n',
    '三工况元数据指向同一`RF0.105_data36QAM.txt`，同一v1/v2与两份PAM4标签哈希完全一致。新工况训练区与同一Tx的符号相关峰在lag=0（513 RX1/RX2约0.964/0.939；382约0.961/0.934），支持复用载荷。没有逐采集独立Tx文件或发射随机种子记录，不能证明仪器级每一比特的独立重发过程。当前证据明确不足以称为不同消息泛化。各模型只拟合各自训练区；509测试标签在过去已被看过，也对应新采集的同一载荷位置，因此跨工况结果仅作条件验证。\n',
    '新工况不是零样本迁移：锁定509选出的架构/超参数后，用每个工况前10000符号独立训练前端、归一化、读出和LMS。新工况验证区没有用于拟合或选模。509前端直接加载第一阶段固定模型；新工况训练诊断均lag=0、RX1/RX2极性+1，与509 RX2极性−1不同，未硬套历史符号。\n',
    table(['区间(1-based)','原始区间','训练/计分区间','复符号数'],[['train','1–10000','129–9872','9744'],['validation','10001–20000','10129–19872','9744'],['test','20001–51200','20129–51072','30944']]),
    '所有方法共享split、两端128保护、单段零填充、独立递归状态与解码。现有保守上下文检查需98符号，小于128；递归记忆并非硬截断，但绝不从另一段取状态。标签不用于初始化NLTCP前驱。每测试条件123776个真实参考bit。居中窗口含未来57符号，是离线处理。全部缩放仅在训练有效窗口拟合；分支状态没有额外标准化。\n',
    '## 3. 搜索规则、消融与验证结果\n',
    '共18个预定义主要配置，两路使用同一个候选配置；没有挑seed。A:两种ridge；B/C:80与160节点×两种ridge；D:40+40与80+80×两种ridge，额外α=0.3/0.9、γL=1、纯状态各1种。其余默认α=0.6、γL=0.7、γN=0.7、ρN=0.9。ridge为1e−5或0.01。\n',
    '按RX1/RX2/XY验证固定2-bit上界的等权平均，在每个A/B/C/D家族内独立选最小值；平局依次比平均MSE、特征维度、名字。这三个输出相关，不被当作三个独立重复试验；等权平均仅是预定工程目标。每家族最优加7个预定预算/直通消融的并集，共9个模型统一最终评价。没有根据新工况结果更换家族代表。\n',
    table(['候选','RX1上界','RX2上界','XY上界','三输出均值','家族选定'],[[r['config']['name']]+[f"{r['validation'][ch]['ber_upper']:.6f}" for ch in ['rx1','rx2','xy']]+[f"{sum(r['validation'][ch]['ber_upper'] for ch in ['rx1','rx2','xy'])/3:.6f}",'是' if r['config']['name'] in names else ''] for r in validation]),
    '![全部候选](validation_search.png)\n',
    '## 4. 冻结测试对照\n',
    '下表先列历史兼容的MATLAB自动位宽BER，随后列完整可解释指标。**自动位宽BER不是固定2-bit物理BER**；尤其382合并结果发生I/Q位宽差异，不能只按第一张表排序。所有工况权重先保存并写哈希，再执行一次统一evaluate；推理后哈希不变。509原RC和LMS测试预测及mask与第二阶段逐元素一致。\n']
    for idx,record in results.items():
        doc += [f'### index={idx}\n',table(['模型','RX1 auto BER','RX2 auto BER','XY auto BER'],[[name]+[fmt(record['metrics'][ch][name]) for ch in ['rx1','rx2','xy']] for name in allnames+['lms']]),
            table(['模型','输出','固定2-bit界限','非法率','非法数','I/Q自动位宽'],[[name,ch.upper(),bounds(record['metrics'][ch][name]),f"{record['metrics'][ch][name]['invalid_rate']*100:.4f}%",record['metrics'][ch][name]['invalid_pam4_decisions'],'/'.join(str(v['width']) for v in record['metrics'][ch][name]['matlab_auto_width'])] for name in names+['lms'] for ch in ['rx1','rx2','xy']]),
            f'![{idx} RX1星座](index{idx}/constellation_rx1.png)\n',f'![{idx} RX2星座](index{idx}/constellation_rx2.png)\n',f'![{idx} XY星座](index{idx}/constellation_xy.png)\n']
    doc += ['### BER解释与不能忽略的反例\n',
    r'保持原解码 $d_t=|q_t-\lfloor q_{t-1}/2\rfloor|$，预测可为非法4/5，不截断、不取模、不强制映射。合法判决比特错误数E，非法分量数U，有效复符号数N：$E/(4N)\le BER_{2bit}\le(E+2U)/(4N)$。非法率为U/(2N)。上下界是未知映射的确定性范围，不是置信区间；全部计数、每分量位宽和MSE见JSON/CSV。'+'\n',
    '382 XY中，双分支auto=0.000291小于tanh80的0.000339，但两者位宽分别3/3和3/2，固定上界分别0.000420和0.000404，排序相反。tanh80固定上界0.000404甚至低于Ridge下界0.000412，说明该单独工况的合并输出确有局部收益；这既不支持双分支普遍优势，也不能推广到其它通道/条件。不得把不同位宽auto数值做统一相对降幅宣传。\n',
    '![固定bit界限](ber_bounds.png)\n',
    '## 5. 预算、时间和特征价值\n',
    table(['模型','线性/tanh节点','输入直通','读出实维','训练实参数','固定随机实参数','λ'],[[name,f"{byname[name]['config']['linear_nodes']}/{byname[name]['config']['nonlinear_nodes']}",str(byname[name]['config']['passthrough']),byname[name]['complexity']['rx1']['feature_dim'],byname[name]['complexity']['rx1']['trained_parameters'],byname[name]['complexity']['rx1']['fixed_parameters'],byname[name]['config']['ridge']] for name in allnames]),
    '40+40双分支与80节点单分支同为311读出维、622训练参数；80+80与160单分支同为391维、782参数。共享输入直通均230维，偏置1维。双分支块对角递归的固定权重数/乘加成本比同总节点的单块更少，但有两次状态循环和不同激活，因此这是匹配总节点及读出维度的对照，**不是严格等计算量**。模型没有可训练的隐藏层参数；共同波形前端不计入表。\n',
    table(['同λ=1e−5模型','总节点','验证均值上界','509测试均值上界','513测试均值上界','382测试均值上界'],[[name,byname[name]['config']['linear_nodes']+byname[name]['config']['nonlinear_nodes'],f"{sum(byname[name]['validation'][ch]['ber_upper'] for ch in ['rx1','rx2','xy'])/3:.6f}"]+[f"{sum(results[idx]['metrics'][ch][name]['ber_upper'] for ch in ['rx1','rx2','xy'])/3:.6f}" for idx in ['509','513','382']] for name in ['tanh80_low','linear80_low','dual40_low','tanh160_low','linear160_low','dual80_low']]),
    '在相同总节点、相同λ的对照中，双分支测试综合上界可以低于对应tanh单分支，但仍高于对应线性RC；且验证区40+40不优于tanh80。这个局部相对收益已完整保留，不代表双分支超越最强基线。\n',
    '下面为509本次单次墙钟秒数：每通道训练10000符号、验证10000、测试31200；均衡器计时不含共同前端或模型磁盘加载。训练包含构造/状态与岭拟合，不含额外诊断。计时不等于实时硬件速度；完整其它工况训练/推理时间也随结果JSON保存。\n',
    table(['模型','通道','训练秒','验证推理秒','测试推理秒'],[[name,ch,f"{byname[name]['complexity'][ch]['fit_seconds']:.5f}",f"{byname[name]['validation_inference_seconds'][ch]:.5f}",f"{results['509']['inference_seconds'][ch][name]:.5f}"] for name in allnames for ch in ['rx1','rx2']]),
    '### 两分支是否提供不同信息\n',
    '只用训练状态拟合投影，在509验证状态上检查残差。选定40+40分支：线性状态无法由当前窗口解释的方差比例仅RX1 0.1014%、RX2 0.1101%；非线性状态的该比例为19.42%/18.50%，在同时给出输入和线性状态后仍为19.37%/18.43%。因此两分支并非数值复制，但线性分支大部分已包含在直通输入中，非线性状态的独特变化也未证明是预测目标所需的信息。线性递归实际上是输入的稳定线性时间滤波，可补充窗口外历史和改变正则化几何，不会凭Identity本身产生非线性。\n',
    '40+40读出名义311维，训练矩阵在相对奇异值阈值1e−10下秩293；80+80名义391维、秩336。选定λ=0.01的正规矩阵条件数约2.93e7/2.45e7，低λ双分支则约2.9e10以上。冗余使条件数恶化。独立SVD核查在所有9个冻结模型/两通道的有效验证区最大预测差7.49e−9；主方法约3.59e−12，支持本次有效区预测的数值可靠性。\n',
    '数值审计也保留了一次失败日志：最初把未评分重置边缘一起检查时，linear80最大差1.19e−5超过临时1e−5检查阈值。随后分别报告全段与既有保护mask，最大全段差6.45e−5，有效区仍≤7.49e−9。没有改动任何模型权重、保护区或测试规则；边缘差异不得隐去。SVD仅为数值核验，正式预测仍来自已冻结原正规方程解。\n',
    '纯状态80+80压缩掉230维直通后，验证综合上界0.3804，三个工况测试也明显失败。这是具体随机状态、预算与参数下的信息保留不足的证据，不能据此证明所有无直通RC都不可能工作。相同λ的预算对照与主模型全部保留；没有将节点增加等同于结构贡献。\n',
    '## 6. 研究判断与下一步\n',
    '**本轮不支持“线性＋非线性RC特征融合稳定优于强线性/单分支”这一假设。** 验证期D家族最优上界均值0.023630，A为0.023160、B为0.023356、C为0.023151。509/513中双分支没有稳定胜过最强线性方法；382 XY有局部差异但auto位宽会改变排序。主双分支对本轮固定参数LMS有改善，不代表优于充分调优的LMS，更不能据此证明双分支设计必要。\n',
    '当前更合适的解释是：长窗口和线性读出已吸收大部分可用信息，线性状态冗余，而新增tanh特征未给出稳定增益。共同前端已有Volterra非线性可能降低后级非线性需求，这是待验证解释而非本轮因果结论。本轮仅一个固定seed、两个额外工况、同一载荷，没有多seed置信区间或独立新数据总体推断。小的误码差不应表述为统计显著。\n',
    '建议下一步优先：\n\n1. **残差/正交化特征融合**：训练区先拟合线性预测及状态投影，再让非线性分支补充残差；用训练尺度标准化与分块Ridge控制冗余，仍保留总维度对照，不引入MLP。\n2. **有依据地改善非线性特征**：固定小规模seed集合和输入增益，检查饱和、残差预测力；对比在共同Volterra前端之后是否还有稳定的非线性残差。不要复用本轮测试选择新配置。\n3. **新载荷、封存采集验证及解码契约**：使用不同Tx随机序列或新采集，事前锁定工况和超参数；与课程方明确非法PAM4的最终接收规则，才报告唯一物理2-bit BER。\n',
    '## 7. 接口、测试与复现\n',
    '`src/dsp_rc/fusion.py`提供`LinearReservoir`、`FusionModel`及18候选。`fit(z_train,y_train,mask)`只拟合训练scale和Wout；`extract(z)`返回input(T,230)、states字典、features(T,F)、final_states；`transform(z)`额外返回prediction(T,)复数；`predict(z)`无标签入口；`save/load`无pickle，存实际矩阵、完整配置、scale、Wout与复杂度。每次extract/transform重置状态，应传完整单段；不能把任意分块当作居中窗口的连续在线处理。\n',
    '18/18 unittest通过（原13项＋新增5项），新增覆盖Identity闭式递推、仿射叠加/收缩、拒绝不稳定参数、状态续接、共同特征维度、只用mask内标签、序列推理不改权重、保存重载一致，以及单tanh预测与Phase2完全一致。小样本只用原训练段的前3000训练/随后1000检查，两个通道端到端通过。Phase3没有另运行MATLAB；Phase2共享矩阵数值验证已复用，新增Identity/融合以独立闭式与SVD检查为证据，不能称为新的MATLAB逐阶段认证。\n',
    '在`/Users/oliver/Desktop/DSP/rc_baseline`运行，现有环境无需安装依赖，输出必须使用新目录：\n',
    '```bash\nexport PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n.venv/bin/python -m unittest discover -s tests -v\n.venv/bin/python -m dsp_rc.phase3 prepare --out runs/phase3_reproduce\n.venv/bin/python -m dsp_rc.phase3 smoke --out runs/phase3_reproduce\n.venv/bin/python -m dsp_rc.phase3 search --out runs/phase3_reproduce\n.venv/bin/python -m dsp_rc.phase3 fit-extra --out runs/phase3_reproduce\n.venv/bin/python -m dsp_rc.phase3_numerics --out runs/phase3_reproduce\n.venv/bin/python -m dsp_rc.phase3 evaluate --out runs/phase3_reproduce\n.venv/bin/python -m dsp_rc.phase3_report --out runs/phase3_reproduce\n```\n',
    '复跑只能称同一数据的复验，不能恢复盲测性质。`prepare`拒绝已有目录，`evaluate`在开始时落盘标记并拒绝重复执行。算法与模型哈希在测试前冻结，推理后核对；报告脚本只读现成预测，未重新训练或调用预测。\n',
    '关键文件：`protocol.json`预声明、`smoke.json`轻量验证、`validation_results.json`全部训练/验证指标、`frozen_algorithm.json`家族选择、`frozen_all_models.json`全部工况模型冻结、`test_results.json`完整测试、`metrics.csv`平表、`feature_diagnostics_validation.json`冗余诊断、`independent_svd_validation_check.json`数值核验、`verification.json`旧文件及预测一致性。每个`index*/`保存模型NPZ、源文件manifest、测试预测(mask及全局符号ID)、星座图；新工况另有前端模型与仅训练区的校准记录。`source_manifest_final.json`登记新增交付源码/测试哈希。未实现可选输出加权融合或MLP，以保持本组共同读出方法的范围。\n']
    report='\n'.join(doc)
    report=re.sub(r'(!\[[^\]]*\]\()([^/][^)]*)(\))',lambda m:m[1]+str(out/m[2])+m[3],report)
    (out/'REPORT.md').write_text(report)
    write_json(out/'source_manifest_final.json',{str(p.relative_to(root)):file_hash(p) for p in [root/'rc_baseline/src/dsp_rc/fusion.py',root/'rc_baseline/src/dsp_rc/phase3.py',root/'rc_baseline/src/dsp_rc/phase3_numerics.py',root/'rc_baseline/src/dsp_rc/phase3_report.py',root/'rc_baseline/tests/test_phase3.py']})
    print('Report, all constellation figures, validation chart, BER bounds and metrics CSV created.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('runs/phase3_feature_fusion'));a=p.parse_args();run(Path(__file__).resolve().parents[3],a.out.resolve())
