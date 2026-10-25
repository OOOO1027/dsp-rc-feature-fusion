"""Phase4 reporting and saved-prediction diagnostics; no model fitting."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .phase3 import read
from .phase4 import aggregate
from .experiment import write_json,file_hash

def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])+'\n'
def image(out,name,alt):return f'![{alt}]({out/name})\n'
def run(root,out):
 rows=read(out/'candidate_results.json');byname={r['config']['name']:r for r in rows};ids={r['config']['name']:f'C{i+1:02d}' for i,r in enumerate(rows)}
 selected=read(out/'selection.json');decision=read(out/'decision.json');seeds=read(out/'seed_results.json');boot=read(out/'paired_block_bootstrap.json');diag=read(out/'residual_diagnostics.json');inventory=read(out/'payload_inventory.json')
 pred=np.load(out/'predictions_seed100.npz');arrays=np.load(out/'residual_diagnostics_arrays.npz');mask=pred['mask'];ys=arrays['ideal_validation'];probes=[]
 # Exact incremental value of corrections; target residuals obtained from saved errors.
 for r in rows:
  c=r['config']
  if c['kind']!='residual' or c['eta']!=1:continue
  for split in ['train','validation']:
   for ch in ['rx1','rx2','xy']:
    base=pred[f"{c['base']}_{split}_{ch}"];new=pred[f"{c['name']}_{split}_{ch}"];correction=(new-base)[mask]
    if ch=='xy':residual=(arrays[f"rx1_{c['base']}_{split}_error"]+arrays[f"rx2_{c['base']}_{split}_error"])[mask]/2
    else:residual=arrays[f"{ch}_{c['base']}_{split}_error"][mask]
    power=float(np.mean(abs(residual)**2));cross=float(np.mean((residual.conj()*correction).real));cp=float(np.mean(abs(correction)**2));gain=(2*cross-cp)/power
    probes.append({'name':c['name'],'base':c['base'],'orth':c['orth'],'split':split,'channel':ch,'residual_power':power,'correction_power':cp,'residual_correction_cross':cross,'incremental_R2_eta1':gain,'delta_MSE_eta1':cp-2*cross})
 write_json(out/'residual_predictability.json',probes)
 with (out/'metrics.csv').open('w') as f:
  cols=['id','name','seed','split','channel','symbol_mse','ber_lower','ber_upper','matlab_compatible_ber','invalid_rate','invalid_pam4_decisions','symbols','I_width','Q_width'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader()
  for r in rows:
   for split in ['train','validation']:
    for ch,m in r[split].items():w.writerow(dict(m,id=ids[r['config']['name']],name=r['config']['name'],seed=100,split=split,channel=ch,I_width=m['matlab_auto_width'][0]['width'],Q_width=m['matlab_auto_width'][1]['width']))
  for r in seeds:
   if r['seed']==100:continue
   for ch,m in r['validation'].items():w.writerow(dict(m,id=ids[r['name']],name=r['name'],seed=r['seed'],split='validation',channel=ch,I_width=m['matlab_auto_width'][0]['width'],Q_width=m['matlab_auto_width'][1]['width']))
 # IQ distributions, ACF, receive amplitude, ideal-region energy.
 for base in ['ridge_low','linear80_low']:
  fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
  for k,ch in enumerate(['rx1','rx2']):
   e=arrays[f'{ch}_{base}_validation_error'][mask];d=diag[ch][base]['validation']
   axes[k,0].hist(e.real,bins=65,density=True,alpha=.5,label='I');axes[k,0].hist(e.imag,bins=65,density=True,alpha=.5,label='Q');axes[k,0].set(title=f'{ch.upper()} residual distribution',xlabel='Target minus prediction',ylabel='Density');axes[k,0].legend()
   for comp in ['I','Q']:axes[k,1].plot(range(1,65),d[comp]['acf_lags_0_to64'][1:],label=comp)
   axes[k,1].axhline(0,c='grey',lw=.6);axes[k,1].set(title='Residual temporal ACF',xlabel='Symbol lag',ylabel='Autocorrelation');axes[k,1].legend()
   for split in ['train','validation']:
    q=diag[ch][base][split]['receive_amplitude_quintiles'];axes[k,2].plot([v['mean_rx_amplitude'] for v in q],[v['residual_mse'] for v in q],'o-',label=split)
   axes[k,2].set(title='Receive-amplitude bins (train cutoffs)',xlabel='Mean CAP magnitude',ylabel='Residual MSE');axes[k,2].legend()
  fig.suptitle(f'{base}: index509, train/validation only');fig.savefig(out/f'residual_{base}.png',dpi=150);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
  for ax,ch in zip(axes,['rx1','rx2']):
   regions=diag[ch][base]['validation']['constellation_regions'];values=np.array([v['mse'] for v in regions]).reshape(6,6).T
   im=ax.imshow(values,origin='lower',extent=(-6,6,-6,6),vmin=.3,vmax=.8,cmap='viridis');ax.set(xticks=[-5,-3,-1,1,3,5],yticks=[-5,-3,-1,1,3,5],xlabel='Ideal I',ylabel='Ideal Q',title=ch.upper()+' residual energy by region')
  fig.colorbar(im,ax=axes,label='Validation residual MSE');fig.suptitle(base+' | region counts stored separately');fig.savefig(out/f'regions_{base}.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');b=byname['linear80_low']
 for ax,metric in zip(axes,['symbol_mse','ber_upper']):
  v=[aggregate(r['validation'],metric)-aggregate(b['validation'],metric) for r in rows]
  ax.barh(list(ids.values()),v,color=['#4477aa' if x<=0 else '#bb6677' for x in v]);ax.invert_yaxis();ax.axvline(0,c='black',lw=.8);ax.set(xlabel='Delta '+metric+' versus linear80',title='All24 predefined configurations');ax.grid(axis='x',alpha=.2)
 fig.savefig(out/'candidate_deltas.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for base in ['ridge_low','linear80_low']:
  for orth in [False,True]:
   rr=[r for r in rows if r['config']['kind']=='residual' and r['config']['base']==base and r['config']['orth']==orth]
   label=base+(' + projected H' if orth else ' + raw H');eta=[0]+[r['config']['eta'] for r in rr]
   for ax,metric in zip(axes,['symbol_mse','ber_upper']):
    baseline=aggregate(byname[base]['validation'],metric);vals=[0]+[aggregate(r['validation'],metric)-baseline for r in rr];ax.plot(eta,vals,'o-',label=label);ax.set(xlabel='eta',ylabel='Delta '+metric+' vs its own base');ax.axhline(0,c='grey',lw=.7);ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Residual correction: eta0 is an explicit baseline');fig.savefig(out/'eta_ablation.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for i,(name,d) in enumerate(boot.items()):
  for j,block in enumerate(['256','512']):
   for k,ax in enumerate(axes):
    m=d[block]['mean_delta'][k];lo,hi=d[block]['CI95'][k];ax.errorbar(i+(j-.5)*.15,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=['#4477aa','#ee8844'][j],label='Block '+block if i==0 else None)
 for k,ax in enumerate(axes):ax.axhline(0,c='grey');ax.set(xticks=[0,1],xticklabels=['Residual eta0.25','Standardized group Ridge'],ylabel=['Delta MSE','Delta upper BER'][k]);ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Exploratory paired moving-block intervals, reused validation');fig.savefig(out/'paired_blocks.png',dpi=160);plt.close(fig)
 # Independent data and decoder questions, not sent to anybody.
 payload='''# 向助教索取／核对的新载荷数据清单\n
目前**有不同Tx参考文件证据，但没有确认完整可用的独立NLTCP载荷测试包**。`data36QAM.txt`与RF0.105参考的SHA-256不同，前40000采样零延迟相关约−0.0068；Sheet1/2明确引用前者，而Sheet3/等高线图引用后者。波形不同也可能来自成形、载频或序列移位，不能仅凭哈希/相关断定原始消息独立。\n
请先核对旧`data36QAM.txt`是否采用另一随机PAM4消息；若是，请补齐与其一一对应的以下项目；若不是，请生成新的随机载荷采集：\n
1. 新payload_id、PAM4生成随机种子/生成脚本、两路完整原始PAM4整数及自然二进制/Gray映射说明；不能仅换index、幅度或文件名。\n2. 编码后完整理想I/Q符号、NLTCP编码器初始状态与终止处理；每个符号和标签精确对应关系，首尾填充/保护序列说明。\n3. 实际发送的Tx波形txt（及可选仪器wfm）、采样率、sps、RRC roll-off/span、载频、相位、归一化、帧结构及采样原点。\n4. 同次采集的RX1/RX2完整原始波形、帧/符号ID，明确每个RX index对应哪个payload_id与Tx哈希。提供采集时间、bias/VPP、通道衰减、硬件设置。\n5. 至少区分开发/校准载荷与最终封存载荷；如使用条件内训练，应事先提供训练校准段与测试段隔离约定。若要验证对新消息的直接泛化，应另测旧载荷训练权重在新载荷上的冻结推理，不能和新载荷内重训混称。\n6. 新载荷与现有509/513/382消息不同的证据，及至少数次独立采集噪声实现；样本数按目标BER与允许不确定度提前约定，不能看到成绩才延长或截短。\n
`data49QAM`、`data36qamQAM`、`pre_SNRest_36QAM`等其它文件的具体编码/用途仍须核对；目前没有各自成套NLTCP标签。`Data/`的64APSK标签属于不同调制任务，不能替代新NLTCP消息的评价真值。\n
本轮止损后未对这些候选文件执行模型测试，也未把不同index当作新消息盲测。此文仅为本地待确认清单，未向助教发送消息。\n'''
 (out/'NEW_PAYLOAD_REQUEST.md').write_text(payload)
 decoder='''# NLTCP非法PAM4输出与课程评价口径待确认事项\n
沿用原函数：先将预测I/Q最近邻判到PAM6编号q∈{0,…,5}，再计算d[t]=abs(q[t]−floor(q[t−1]/2))。前/当前q为(0,4)、(0,5)、(1,4)、(1,5)、(2,5)、(3,5)时会输出4或5。均衡误差改变当前或前驱PAM6判决会产生非法转换；不能当作正常PAM4符号强行截断或取模。\n
当前保持：合法判决按自然二进制精确计错；非法分量数U，每个未知2-bit输出可能贡献0…2错。N个复符号、合法比特错误E，固定2-bit BER范围为[E/(4N),(E+2U)/(4N)]，非法率U/(2N)。这不是置信区间，也不是对一个已定义合法解码器的唯一实测BER。MATLAB自动位宽BER另列：出现4/5时可能扩大为3位，I/Q位宽可不同，分母不再等于发射端实际比特数。\n
请助教明确：\n
1. 原有记忆解码公式是否是课程指定的最终接收算法？4/5应视为擦除、整符号错误、丢帧，还是应采用约束序列判决？若要求合法输出，请提供明确状态转移、代价、初态和终态，不应由我们依据测试标签选择映射。\n2. 最终BER的分母是否固定每PAM4两bit、每复符号四bit？擦除对应两个bit是否都计错，还是另外报告擦除率？若允许剔除，必须同时报告覆盖率及原分母，不能通过丢弃坏点降低BER。\n3. PAM4整数到bit是自然二进制还是Gray，I/Q如何复用？现有文件/代码使用自然二进制比较，需课程方确认。\n4. 源标签首项缺前驱、末项与理想解码不一致如何定义？此前核对第1和51200项不一致；是否应丢保护区、补齐真实编码状态或修正参考文件？不要在测试中按答案重置解码器。\n5. 当前/前驱判决错误造成的传播是否按解码后的逐bit结果计数？约束解码器如新增，允许的训练与验证范围、延迟、输出时序如何统一？\n6. 各方法是否必须用同一同步、固定统计mask、同一解码器及边界规则？RX1/RX2与等权合并应分别解码后评价，避免混用位宽或统计区间。\n
在书面规则确认前继续报告上下界、非法率和历史兼容BER。若今后改规则，只能对已保存预测作明确标注的历史重评分；不能把它称为新盲测，也不能依据旧测试重评分继续选方案。此文未对外发送。\n'''
 (out/'NLTCP_DECODER_QUESTIONS.md').write_text(decoder)
 # Chinese research report, all numerical tables sourced from saved JSON.
 doc=['# DSP第四阶段：残差驱动的双分支RC特征融合研究\n',
 '**结论：本轮未发现稳定的非线性残差预测增益，已按预声明规则止损。** 在两种线性基础×两种状态处理的四组残差方案中，验证选择均回到eta=0。最优非零修正在三种固定非线性种子下都略微增加MSE和固定2-bit BER上界；分组Ridge也没有超过线性RC。不继续扩大搜索，不调用509/513/382旧测试区，不宣称新模型获胜。\n',
 '已完成24个主要配置、2个固定设计×3种子诊断、23项单元测试、小样本接口验证、残差结构图和成套模型/指标留档。前三阶段源码/模型/结果保留。研究限于NLTCP index509的原训练与验证区，所有“验证”结果均属开发证据，不是独立新载荷测试。\n',
 '## 1. 协议与数据边界\n',
 '完整阅读并复用第三阶段REPORT、fusion.py、协议、18候选训练/验证结果和全部最终消融；这些历史结果用于确认基线和提出研究问题，没有作为本轮模型选择的数据区。沿用已核验的波形前端、CAP/RRC、sps=4、115复符号居中窗口、NLTCP解码与评分函数。共同前端已经含Volterra非线性，“线性基础”指符号均衡器，并非整条接收链全线性。\n',
 table(['分段','全局1-based区间','真正拟合/计分区间','有效复符号'],[['训练','1–10000','129–9872','9744'],['验证','10001–20000','10129–19872','9744'],['历史测试','20001–51200','本轮不处理/不评分','0']]),
 '每段独立预处理与重置储备池；128符号双端保护、115窗口含未来57符号，仍是离线S2S。加载器按整文件哈希校验后，立即丢弃第20000符号之后的数据，实验函数仅接受两段训练/验证输入。旧测试文件只可能参与完整性哈希核验，未作模型预测或计分。所有scale、均值/方差、投影矩阵与读出仅由训练mask拟合。验证标签用于选择eta和比较方案；未回填训练。\n',
 '候选与止损规则在运行前写入protocol.json。训练区3000符号/随后1000符号的smoke只验证接口与持久化；其共同前端来自原完整训练区，故不称为独立小样本泛化实验。正式23项测试另覆盖训练标签mask、训练尺度隔离、预测不改权重、eta=0和等价变换。\n',
 '## 2. 线性模型残差的结构\n',
 '分析两种基线：输入窗口Ridge与Identity线性RC80。残差为目标减预测，均使用原理想符号I/Q尺度。下表为验证区；训练区全部统计、64阶ACF、分位数、偏度/峰度、状态相关矩阵均保存在residual_diagnostics.json。\n',
 table(['基线','通道','I均值 / 方差','Q均值 / 方差','I/Q lag1 ACF','幅度相关'],[[base,ch,f"{diag[ch][base]['validation']['I']['mean']:.5f} / {diag[ch][base]['validation']['I']['variance']:.5f}",f"{diag[ch][base]['validation']['Q']['mean']:.5f} / {diag[ch][base]['validation']['Q']['variance']:.5f}",f"{diag[ch][base]['validation']['I']['acf_lags_0_to64'][1]:.4f} / {diag[ch][base]['validation']['Q']['acf_lags_0_to64'][1]:.4f}",f"{diag[ch][base]['validation']['corr_receive_amplitude_residual_abs']:.4f}"] for base in ['ridge_low','linear80_low'] for ch in ['rx1','rx2']]),
 '残差均值接近0但不是完全白噪声；RX1一阶负相关约−0.13/−0.14，RX2更弱。残差绝对值与接收CAP幅度相关约0.07–0.08，说明只有弱幅度依赖。幅度分箱阈值完全来自训练区，验证沿用；不能把这些相关直接等同于可预测非线性。\n',
 image(out,'residual_ridge_low.png','窗口Ridge残差分布、自相关与幅度分箱'),image(out,'residual_linear80_low.png','线性RC残差诊断'),
 table(['基线','通道','外层符号比例','外层残差能量占比','外层/内层MSE','最大状态相关 I/Q'],[[base,ch,f"{diag[ch][base]['validation']['outer_level_region']['symbol_fraction']:.2%}",f"{diag[ch][base]['validation']['outer_level_region']['error_energy_fraction']:.2%}",f"{diag[ch][base]['validation']['outer_level_region']['mse_outer']:.5f} / {diag[ch][base]['validation']['outer_level_region']['mse_inner']:.5f}",' / '.join(f'{v:.4f}' for v in diag[ch][base]['validation']['max_abs_state_corr_per_IQ'])] for base in ['ridge_low','linear80_low'] for ch in ['rx1','rx2']]),
 '外层定义为理想I或Q绝对值等于5。它占约23.4%的符号，贡献约28%的残差能量，存在温和的高电平误差集中；并非错误几乎全在高幅度区。按真值星座分组只用于诊断，不作为接收端可用输入，也不据此加权训练或后处理。单状态最大相关约0.026–0.034，是80维中取最大值，未经多重比较修正，更不能当作预测价值证据。\n',
 image(out,'regions_ridge_low.png','Ridge星座区域残差能量'),image(out,'regions_linear80_low.png','线性RC星座区域残差能量'),
 '## 3. 残差模型、近似正交化和分组惩罚\n',
 '两分支独立固定矩阵：Identity80（seed200、A算子范数0.6、leak0.7）与tanh80（主seed100、A谱半径0.9、leak0.7）。线性稳定性继续使用第三阶段的有效递归收缩证明。输入230实维按训练最大绝对值缩放。设X为窗口、L为线性状态、H为非线性状态，C=[X,L,1]，F=[C,H]。\n',
 r'$$W_B=\operatorname{Ridge}(B_{tr},Y_{tr};10^{-5}),\quad R_{tr}=Y_{tr}-B_{tr}W_B,\quad B=[X,1]\text{ 或 }[L,X,1].$$'+'\n',
 r'$$P=(C_{tr}^TC_{tr}+0.01I)^{-1}C_{tr}^TH_{tr},\quad E=H-CP,\quad V=[(E-\mu_{tr})/\sigma_{tr},1].$$'+'\n',
 '原始状态残差方案令P=0；投影方案P仅拟合一次训练区再用于验证，投影系数0.01固定不搜索。由于有Ridge项，这是近似正交化/状态残差化，CᵀE一般不等于0，不声称严格OLS正交。非线性状态逐列标准化，常数列单独处理。\n',
 r'$$W_R=\operatorname{Ridge}(V_{tr},R_{tr};0.01),\quad \hat Y=B W_B+\eta V W_R,\quad\eta\in\{0,0.25,0.5,1\}.$$'+'\n',
 '线性基础保持冻结，不与修正层共同重新拟合。Ridge基础原始状态方案的线性RC状态虽在公共bank中物化，但不进入预测；它是“Ridge＋非线性修正”控制，不冒充两分支都有效参与。线性RC基础才是两分支都直接参与的残差模型；Ridge基础的投影版本也使用线性状态来定义H的残差。\n',
 '### 不能把坐标变换包装成新能力\n',
 r'$$\widetilde F=FT,\quad W=T\widetilde W.\quad \|FW-Y\|^2+\lambda\|W\|^2=\|\widetilde F\widetilde W-Y\|^2+\lambda\|T\widetilde W\|^2.$$'+'\n',
 '训练均值/标准差构成可逆仿射变换（偏置吸收平移）；H→H−CP也是可逆块三角坐标变换。同步将惩罚因子设为sqrt(0.01)T后，模型函数空间与原直接拼接Ridge完全相同。本轮两个equivalent控制在真实验证区预测最大差约1.18e−11，BER完全一致，因此明确不把它们计作能力提升。\n',
 '若变换后仍使用对角惩罚，则等效于改变原坐标中的正则先验，而不是新增信息。uniform在标准化坐标对所有系数用0.01；group分别给输入0.01、线性状态1、非线性状态10、偏置0.01。std_uniform→std_group只改变分组惩罚；std→orth后再标准化且保持对角惩罚，同时改变了坐标与诱导正则，不能单称为正交化本身贡献。残差目标则另改变了训练次序、冻结约束和eta收缩。若无正则、精确OLS投影、eta=1，部分两步残差回归与联合OLS可等价；本实验并不满足这些条件，不把一般重参数化误认为新非线性能力。\n',
 '上述坐标等价结论限于保留完整C且全部读出系数可训练的联合模型；顺序残差模型的基础读出冻结，不能自动吸收−CP带来的线性修正，因此不能直接套用同一等价证明。残差方案与标准化联合方案也不是仅改目标的单因素对照：基础λ=1e−5且冻结，联合方案在变换坐标使用0.01/分组惩罚。其差异只能归于整体训练约束与正则化组合，不能全部归功于残差目标。可单独归因的对照是固定修正层的eta变化、同坐标的uniform/group变化，以及同步变换惩罚的等价控制。\n',
    '新的修正/联合读出通过增广QR最小二乘解[训练特征;惩罚因子]，避免再平方条件数；强线性参考仍调用既有FusionModel保持历史一致。QR数值改善没有被当作模型增益。\n',
 '## 4. 24个预定义配置与全部验证对照\n',
 '主seed100，线性seed200；固定输入、拆分和前端，不搜索窗口、隐藏节点数或输入增益。6个既有参考＋12个残差配置（2基础×2状态处理×3正eta）＋6个联合/等价控制=24。四个eta=0是基线别名，不额外训练。每通道共用同一结构/eta，最后RX1/RX2固定0.5/0.5合并。\n',
 '选择目标首先是三输出复数MSE等权平均，再比BER上界、小eta、名称；三输出相关，这只是工程加权指标，并非三个独立试验。完整train/validation、每通道指标和所有seed在metrics.csv中。下面没有省略任何主要候选。\n',
 table(['ID','配置','验证MSE均值','验证上界均值','验证auto均值','非法率均值'],[[ids[r['config']['name']],r['config']['name'],f"{aggregate(r['validation'],'symbol_mse'):.9f}",f"{aggregate(r['validation'],'ber_upper'):.9f}",f"{aggregate(r['validation'],'matlab_compatible_ber'):.9f}",f"{aggregate(r['validation'],'invalid_rate'):.6%}"] for r in rows]),
 image(out,'candidate_deltas.png','24候选对最强线性RC差值'),image(out,'eta_ablation.png','eta含零消融'),
 '四组eta选择结果：\n',table(['残差家族','包括eta=0后的选择'],list(selected['residual_family_winners_including_eta0'].items())),
 '### 主要方法逐通道指标\n']
 report_names=['ridge_low','linear80_low','linear160_low','tanh80_low','dual40_reg','dual80_reg',selected['shortlisted_positive_residual']['name'],'res_ridge_low_raw_eta0.25','res_linear80_low_orth_eta0.25','joint_std_uniform','joint_std_group','joint_orth_group']
 doc+=[table(['ID','输出','MSE','固定2-bit下界–上界','历史auto BER','非法率'],[[ids[name],ch,f"{byname[name]['validation'][ch]['symbol_mse']:.6f}",f"{byname[name]['validation'][ch]['ber_lower']:.6f}–{byname[name]['validation'][ch]['ber_upper']:.6f}",f"{byname[name]['validation'][ch]['matlab_compatible_ber']:.6f}",f"{byname[name]['validation'][ch]['invalid_rate']:.4%}"] for name in report_names for ch in ['rx1','rx2','xy']]),
 '## 5. 非线性状态究竟有没有额外预测价值\n',
 '直接考察修正层预测真实线性残差的能力，而不是比较两个状态矩阵是否不同。下表取eta=1，R²定义为1−均衡后残差MSE/基础残差MSE，不减去验证残差均值；这是相对于“零修正”的增量R²。正值代表改善，负值代表恶化。\n',
 table(['基础','状态','输出','训练增量R²','验证增量R²'],[[base,'近似正交' if orth else '原始',ch,f"{next(p['incremental_R2_eta1'] for p in probes if p['base']==base and p['orth']==orth and p['channel']==ch and p['split']=='train'):.6f}",f"{next(p['incremental_R2_eta1'] for p in probes if p['base']==base and p['orth']==orth and p['channel']==ch and p['split']=='validation'):.6f}"] for base in ['ridge_low','linear80_low'] for orth in [False,True] for ch in ['rx1','rx2','xy']]),
 r'对任一固定修正 $\hat r$，$\Delta\mathrm{MSE}(\eta)=\eta^2E|\hat r|^2-2\eta\Re E[r^*\hat r]$。`residual_predictability.json`保存这两个项，避免把很小的残差/状态相关夸大成预测收益。训练可解释部分增加而验证增量R²为负，符合有限样本额外特征拟合训练扰动的现象；不能由此证明所有非线性信息不存在。'+'\n',
 '正交化使修正不再主要复述输入/线性状态，但也未带来验证收益。主seed下最好的非零修正是线性RC基础＋原始标准化H、eta=0.25，其平均MSE仍比基础增加0.00002909。eta=0在本预定义网格中胜出；没有据此再试更小eta、更多节点或更多seed追求低BER。\n',
 '## 6. 固定多种子、配对时序重采样与止损\n',
 '按协议对最优非零残差和最优非等价联合设计，固定结构、lambda和eta，检查非线性seed100/101/102；线性seed200保持不变，专门检查非线性初始化敏感性。不选择表现最好的seed。这是额外初始化诊断，不是新数据独立重复；也没有覆盖线性储备池的种子方差。\n',
 table(['设计','非线性seed','验证MSE均值','验证上界均值'],[[r['name'],r['seed'],f"{aggregate(r['validation'],'symbol_mse'):.9f}",f"{aggregate(r['validation'],'ber_upper'):.9f}"] for r in seeds]),
 '主协议要求三seed均降低MSE、平均至少0.1%增益、无seed恶化BER上界，并且两种块长的配对区间均支持两指标下降。门槛只是事前工程止损标准，不是普适显著性定义。两个设计均不通过：\n',
 table(['设计','三seed平均MSE相对收益','通过门槛'],[[name,f"{v['relative_mean_mse_gain']:.6%}",str(v['pass_gate'])] for name,v in decision['methods'].items()]),
 '连续256/512符号移动块，各2000次、固定seed404。先在完整验证序列解码出每符号MSE和BER上界贡献，再对相同位置块配对重采样，不将拼接后的块重新解码，以免引入假边界；所有通道和种子使用相同索引。先对seed损失差取均值，保留通道相关性。块长超过115窗口，属于两种预定敏感性检查，不保证完全消除所有长程相关。\n',
 table(['设计','块长','ΔMSE 95%区间','ΔBER上界 95%区间'],[[name,b,' / '.join(f'{x:.8g}' for x in v['CI95'][0]),' / '.join(f'{x:.8g}' for x in v['CI95'][1])] for name,blocks in boot.items() for b,v in blocks.items()]),
 image(out,'paired_blocks.png','配对连续块重采样'),
 '差值为新方法减线性RC，负值才有利。非零残差修正的MSE区间跨0，BER上界区间偏正；分组联合读出的MSE区间明显偏正。**这些是反复使用的验证集上、选模之后的探索性区间，没有做选择偏差/多重比较校正，不能作为独立确认性置信声明。** 即使不依赖区间，三seed原始点估计也未满足止损条件。\n',
 '## 7. 参数预算、运行时间与复用接口\n',
 '公共80+80 bank含49760个固定随机实系数。输入Ridge为231维读出，linear80/tanh80为311，linear160/dual80为391；dual40也是311。残差模型另有81维修正（80状态＋偏置）：Ridge基础总读出系数624，linear80基础784；与联合80+80的782接近，但两步优化约束不同、偏置重复，不能称完全等参数。Ridge原始状态修正的线性bank不影响输出，属于控制方案。\n',
 '投影矩阵311×80=24880个**训练拟合的无监督系数**，不能漏计成固定随机参数；均值/标准差也训练估计。表中将它们与目标读出参数分列。等价控制存储稠密变换T只是推导出来的表示，不是新增自由参数。推理当前残差实现复用base.predict，线性基线状态可能重复计算；没有宣称计算最优或实时性。\n',
 table(['ID','目标读出参数','投影拟合系数','RX1训练/推理秒','RX2训练/推理秒'],[[ids[r['config']['name']],r['stats']['rx1']['readout_parameters'],r['stats']['rx1']['projection_fitted_coefficients'],f"{r['stats']['rx1']['fit_seconds']:.5f} / {r['inference_seconds']['rx1_validation']:.5f}",f"{r['stats']['rx2']['fit_seconds']:.5f} / {r['inference_seconds']['rx2_validation']:.5f}"] for r in rows]),
 '每次训练/推理段均10000符号，评分9744；计时为本机单次墙钟、不含共同前端与磁盘保存。搜索中的特征bank初始化复用，bank_setup_seconds另存；残差训练包含重拟合其线性基础。额外seed没有复用bank，时间不能直接解读为相同工作量基准。\n',
 '`src/dsp_rc/residual_fusion.py`提供FeatureBank、ResidualFusion、coordinate_transform与penalized_lstsq。fit仅接收训练z/y/mask；components(z)返回基础预测及单位eta修正；predict(z,eta=None)按配置融合，无标签入口；save/load保存实际矩阵、投影、尺度、读出和配置。所有候选/seed的重载预测逐元素一致。正式开发的eta=0优胜意味着部署选择现有线性基线，不能把非零诊断候选误称为胜出模型。\n',
 '## 8. 新载荷证据与NLTCP解码规则\n',
 '本地发现不同Tx参考文件，不能笼统说“只有一个Tx文件”：\n',
 table(['Tx txt','样本形状','与RF参考训练前缀相关'],[[Path(v['path']).name,str(v.get('shape','')),'—' if 'train_prefix_correlation_RF0105' not in v else f"{v['train_prefix_correlation_RF0105']:.6f}"] for v in inventory['tx_files'] if v['kind']=='.txt']),
 '早期Sheet1/2指向data36QAM.txt，后期Sheet3/等高线图指向RF0.105_data36QAM.txt；前者不同哈希且零延迟相关很低，是不同参考波形的明确证据，但仍不能排除不同成形/载频/移位，更不能在缺乏对应标签时称为独立消息。仅发现一套v1/v2与decimal_PAM4_1/2，尚无可验证的另一组完整NLTCP载荷包。Data目录64APSK是另一任务。没有拿这些不完整候选试模型、挑结果。\n',
 f'具体数据请求见[NEW_PAYLOAD_REQUEST.md]({out/"NEW_PAYLOAD_REQUEST.md"})：要求payload_id/随机种子、两份原PAM4消息、对应理想符号、编码状态、Tx成形参数、RX1/RX2及逐帧配对元数据。需要先核实旧data36QAM是否确为另一消息，能补齐则可另立验证协议；否则向助教索取新随机载荷采集。本轮方法未过门槛，暂不进入最终新载荷性能宣称。\n',
 f'非法PAM4规则单独整理于[NLTCP_DECODER_QUESTIONS.md]({out/"NLTCP_DECODER_QUESTIONS.md"})。维持原解码4/5，不截断、不取模、不按答案映射。固定2-bit范围为[E/(4N),(E+2U)/(4N)]，非法率U/(2N)；这只是未定义映射的确定性界限，不是物理BER精确值或置信区间。历史auto逐I/Q按最大整数定宽。每方法同一mask；完整位宽和计数在CSV/JSON。向助教确认编码/解码状态、非法输出处理、自然/Gray映射、分母、擦除计错和首尾标签。未发送任何外部消息。\n',
 '## 9. 结论、限制与复现\n',
 '**当前不值得继续扩大这一固定前端、115窗口、随机tanh状态的搜索。** 第四阶段的价值在于将“状态不同”与“残差可预测”分开，并实测坐标等价控制。当前非线性状态对训练残差有拟合作用，却没有在预定验证网格及三seed检查中产生稳定增量；保持eta=0、保留强线性基线是本轮明确决策。\n',
 '限制包括单工况、同一反复用于研究的验证区、一个线性seed、三个非线性seed、有限eta/正则网格、共同前端已含Volterra、训练残差来自样本内拟合，以及未解决的非法PAM4输出。不能从本轮否定一切非线性方法或任意更小eta；也不能因个别通道BER下降挑选局部结论。若后续另有物理信道机制依据，可事前定义不同特征/训练内交叉拟合研究，但本轮不继续追参数。优先补齐新载荷与解码契约。\n',
 '复现：在`/Users/oliver/Desktop/DSP/rc_baseline`使用已有环境，新输出目录必须不存在。以下只复验开发结果，不会恢复独立盲测性质。\n',
 '```bash\nexport PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n.venv/bin/python -m dsp_rc.phase4 prepare --out runs/phase4_reproduce\n.venv/bin/python -m unittest discover -s tests -v > runs/phase4_reproduce/unit_tests_final.log 2>&1\n.venv/bin/python -m dsp_rc.phase4_diagnostics --out runs/phase4_reproduce\n.venv/bin/python -m dsp_rc.phase4 smoke --out runs/phase4_reproduce\n.venv/bin/python -m dsp_rc.phase4 search --out runs/phase4_reproduce\n.venv/bin/python -m dsp_rc.phase4 seeds --out runs/phase4_reproduce\n.venv/bin/python -m dsp_rc.phase4_verify --out runs/phase4_reproduce\n.venv/bin/python -m dsp_rc.phase4_report --out runs/phase4_reproduce\n```\n',
 '交付：protocol.json预声明；residual_diagnostics.json与图表；candidate_results.json完整24配置；selection.json含eta=0与等价检查；seed_results.json与paired_block_bootstrap.json；decision.json止损；models/全部训练模型；predictions_seed100.npz与seed_validation_predictions.npz仅训练/验证预测；metrics.csv全指标；payload_inventory.json原文件哈希；verification.json保护/测试核验；source_manifest.json新增源码。没有Phase4 test预测文件。\n']
 conditioning=read(out/'conditioning.json')
 position=doc.index('## 8. 新载荷证据与NLTCP解码规则\n')
 doc[position:position]=['### 条件数改善不等于预测增益\n',table(['设计','RX1增广矩阵κ₂','RX2增广矩阵κ₂'],[[name,f"{conditioning['rx1'][name]['augmented_design_condition_2']:.2f}",f"{conditioning['rx2'][name]['augmented_design_condition_2']:.2f}"] for name in ['raw_dual80_reg','joint_std_uniform','joint_std_group','joint_orth_uniform','joint_orth_group','joint_std_equivalent','joint_orth_equivalent']]),'正规矩阵条件数是表中数值的平方。分组惩罚将增广矩阵条件数从约6300–6500降至约480–565，但验证性能仍较差；不能把数值条件改善解释为获得非线性信息。近似正交化后训练H剩余方差比例为RX1 18.456%、RX2 17.441%，归一化||CᵀE||约4.13e−7/4.71e−7；投影确实去除了线性可解释部分，仍未产生验证残差增益。\n']
 (out/'REPORT.md').write_text('\n'.join(doc));pred.close();arrays.close()
 verification=read(out/'verification.json');verification['figure_count']=len(list(out.glob('*.png')));write_json(out/'verification.json',verification)
 write_json(out/'source_manifest.json',{str(p.relative_to(root)):file_hash(p) for p in [root/'rc_baseline/src/dsp_rc/residual_fusion.py',root/'rc_baseline/src/dsp_rc/phase4.py',root/'rc_baseline/src/dsp_rc/phase4_diagnostics.py',root/'rc_baseline/src/dsp_rc/phase4_report.py',root/'rc_baseline/tests/test_phase4.py',root/'rc_baseline/src/dsp_rc/phase4_verify.py']})
 print('Chinese report,7 figures, all metrics and separate request/decoder notes saved.',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('runs/phase4_residual_fusion'));a=p.parse_args();run(Path(__file__).resolve().parents[3],a.out.resolve())
