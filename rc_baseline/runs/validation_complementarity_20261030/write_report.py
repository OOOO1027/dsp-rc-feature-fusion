from pathlib import Path
import json,csv
p=Path(__file__).resolve().parent
s=json.loads((p/'summary.json').read_text());m=json.loads((p/'metrics.json').read_text());boot=json.loads((p/'paired_block_bootstrap.json').read_text())
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
doc='''# 验证预测的误差互补分析（index=509）

## 结论与行动判断

**存在少量互相救错，但尚未观察到稳定、可识别的分工。现有证据不支持立即启动“长记忆线性＋短时非线性”的训练实验，更不支持扩大搜索。**

两路残差高度相关，几乎所有 PAM6 判决相同；少量差异集中在判决边界附近。接收幅度、逐符号变化和当前线性判决裕量可以描述出错位置，但没有形成跨 RX1/RX2/XY 一致的“哪一路更可靠”规律。这个结论只针对当前输出和检查的几个简单描述量，不排除更复杂的可学习规律，也不构成新双时间尺度结构的直接失败实验。

若课程必须继续争取性能提升，长/短时间尺度分工仍可作为**尚未验证的新假设**，但其理由应是记忆需求与非线性响应的机制，而不是把本次少量救错宣称为结构有效的证据。本轮不开展该实验。

## 1. 输入、范围与读取边界

- 只分析本组 Phase3 的 `validation_predictions.npz`，index=509；不混入另两名队员不同链路/调制的数据。
- 验证序列全局零基 ID 10000–19999；前后各保护128符号；实际评分 ID 10128–19871，共 **9744** 个复符号，即 PAM4 对。
- 只读取 Phase4 诊断缓存中的 `ideal_validation`、两路 `validation_input` 和 mask；不读取训练预测成员。
- 原始 PAM4 标签正式分析采用无缓冲逐字节 token 流，跳过前10000个值、只解析10000–19999，到第20000个值结束。
- **读取事故披露**：初次使用按行定位的读法，原始 PAM4_1 文件却将全部标签放在同一行，因而读到包含测试后缀的整行文本，随后因未取得10000行触发断言退出。未解析测试后缀、未形成任何测试标签数组或测试指标。确认文件格式时亦读了首行文本。随后改用上述流式读取。不能声称本轮绝对没有读到任何测试后缀字节；可以确认：没有打开测试预测文件、没有使用测试标签进行本次统计/选择、没有计算测试成绩。
- 现有验证区已被多轮使用，所有分层与区间均为探索性诊断；不是新的盲测。
- 只拥有这组单分支的一个固定种子（线性200、tanh100）；其他节点/正则对照不是多种子复制。

## 2. 对照与术语

主对照：`linear80_low` vs `tanh80_low`。两者均115复符号窗口、80节点、输入直通、同一训练/验证划分、同一lambda=1e-5和前端。线性norm=.6、leak=.7；tanh radius=.9、leak=.7。它们是**独立训练的单分支读出**，不是把dual40共同读出任意分割出来的两种“独立预测”。因此若将两套80节点全部融合，总节点会是160，不自动构成与单支80等预算比较。

固定敏感性检查：160节点同lambda，以及80节点lambda=.01。仅分析预先指定配置，无候选选择。`ridge_low`、`dual40_low`、`dual40_reg`作为已有输出背景对照。

RX1/RX2是物理通道；XY指先各自预测，再固定0.5/0.5平均。它与“线性/非线性RC分支融合”不同。XY的接收幅度描述量也采用两路现存CAP输入的等权平均，不代表某个独立物理接收器。

## 3. 两个层面的正确性

1. 预解码：最近PAM6网格的I/Q对是否与理想复符号一致。
2. 主分析：先在各自完整验证序列独立运行现有NLTCP解码器，再判断两个PAM4标签是否**都正确**。任一非法4/5均视为该对错误，并单独记录非法率，绝不截断/取模。

主分析表不是BER表：一个错误PAM4对可能包含不同数量的错bit。解码器依赖前一个PAM6判决，所以相邻错误可能成对出现。所有解码在评分mask之前执行，保护区正常预热，不使用真值初始化解码状态。

'''
doc+='## 4. 主对照四格表\n\n'+table(['通道','两路都对','非线性救回线性','线性救回非线性','两路都错'],[[r['channel'].upper(),r['both_correct'],r['nonlinear_rescues_linear'],r['linear_rescues_nonlinear'],r['both_wrong']] for r in s[:3]])
doc+='\n'+table(['通道','线性错误对数','非线性错误对数','线性错误中被非线性救回','错误集合Jaccard','相同PAM6判决'],[[r['channel'].upper(),r['linear_wrong'],r['nonlinear_wrong'],f"{r['rescue_fraction_of_linear_errors']:.2%}",f"{r['error_jaccard']:.2%}",f"{r['same_pam6_pair_fraction']:.2%}"] for r in s[:3]])
doc+='''
在至少一路出错的位置，约81%–86%是两路共同错误。I/Q连续残差相关系数约0.9946–0.9965；输出并非完全相同，但高度相似。

**理想选择参考（不可部署）**：假如有真值在已经分别解码的两路输出间选择，RX1/RX2/XY错误对数可分别从线性755/900/304降至707/812/270，即线性错误减少6.36%/9.78%/11.18%。它仅是“两路已解码结果逐位置选择”的诊断极限，不是所有融合方式的普遍上界。连续特征融合有可能改正两路都错的位置，也可能破坏两路都对的位置。

NLTCP带前一符号状态：在原始复输出间切换后重新解码，不等于从已分别解码的结果里任选。本分析没有生成或声称可部署的oracle BER。

![互补性与理想选择诊断](complementarity.png)

## 5. BER口径及已有模型背景

固定2-bit界限：每个非法PAM4分量的未知误码数取[0,2]，即[Elegal/(4N),(Elegal+2U)/(4N)]。这是非法定义不确定性界限，不是置信区间。历史autoBER仍按I/Q分别自动位宽后平均；不能与上述固定bit口径混用。

'''
rows=[]
for ch in ['rx1','rx2','xy']:
 for name in ['ridge_low','linear80_low','tanh80_low','dual40_low','dual40_reg']:
  a=m[name+'_'+ch];rows.append([ch.upper(),name,f"{a['ber_lower']:.7f}–{a['ber_upper']:.7f}",f"{a['matlab_compatible_ber']:.7f}",f"{a['invalid_pam4_decisions']/(2*a['symbols']):.4%}",f"{a['symbol_mse']:.6f}"])
doc+=table(['通道','模型','固定2-bit BER界限','历史auto BER','非法率','MSE'],rows)
doc+='''
这些指标复算后与 Phase3 保存的验证指标一致。XY的强线性窗口Ridge在本组表里仍优于两种单支的BER上界和已保存双分支；不能把线性RC与tanh之间的理想选择空间，直接转成超过最强线性对照的保证。

## 6. 时序与可观测分层

- **判决边界**：主对照的全部互救事件均位于线性输出“当前/前一符号最小判决裕量”的最低20%分位。它能找出容易翻转的区域，但不能决定翻向哪一路才正确。
- **接收幅度**：没有一致的“高幅度更应该信非线性”模式。只在恰有一路正确的位置上，用幅度区分非线性救错与线性救错的描述性AUC为RX1=.509、RX2=.496、XY=.498，接近无排序能力。
- **快速变化**：一阶变化幅度的对应AUC为.547/.522/.419，方向不一致；RX2最高分位有26次非线性救错、23次线性救错，但RX1相同分位是12对13，不能只挑RX2这一格宣称有效。
- **线性裕量**：对应AUC=.500/.471/.414；XY的反方向提示仅来自64个互救位置，不作门控阈值或模型选择。这些AUC不是训练/验证分类器成绩，更不是证明所有特征都不可学习。
- **时间区段**：RX1前两四分位非线性救错略多(19:16、15:14)，后两段转为劣势(3:19、11:17)；RX2只有第二四分位略占优。XY第三四分位反而6:10。没有稳定时间段分工。
- **星座区域**：理想外层符号有2280个。外层互救（非线性:线性）为RX1 10:17、RX2 24:23、XY 9:5；内层为38:49、64:76、25:25。它是依赖真值的解释，不能直接作为推理门控。外层信号也没有一致净收益。
- **连续事件**：非线性救错run的中位长度均为1，最长均为2个符号；未观察到长段非线性独占优势。NLTCP自身的一步记忆可使短事件相邻，不能把两符号run解释为信道物理记忆长度。

![预先指定的可观测分层](observable_strata.png)

![连续块中的救错与反向损失](temporal_rescue.png)

## 7. 不确定性及固定敏感性检查

采用256/512符号移动块配对bootstrap（2000次），保留块内相关性。区间是当前重复使用验证记录上的探索性区间，不是跨工况泛化区间。比较nonlinear−linear，正值更差。

'''
rows=[]
for r in boot:
 if r['pair']=='linear80_low__tanh80_low' and r['block']==512:
  rows.append([r['channel'].upper(),r['endpoint'],f"{r['mean']:+.7f}",f"[{r['ci95'][0]:+.7f}, {r['ci95'][1]:+.7f}]"])
doc+=table(['通道','指标差','均值','512符号块95%区间'],rows)
doc+='\n256与512块长下，主对照三个通道的错误对率、BER上下界差区间均跨零；MSE差区间均为正，即tanh的连续预测误差更高。**不能把判决附近少数翻转解释为整体非线性拟合改善。**\n\n'
doc+=table(['固定对照','通道','线性错对','tanh错对','非线性救回','线性救回'],[[r['pair'],r['channel'].upper(),r['linear_wrong'],r['nonlinear_wrong'],r['nonlinear_rescues_linear'],r['linear_rescues_nonlinear']] for r in s[3:]])
doc+='''
增到160节点确实增加分歧，但RX1/RX2的反向损失仍更多；另一档正则化下结论也相似。它们仅作固定敏感性检查，不选择其中某一配置来宣布成功。

## 8. 是否值得开展长记忆线性＋短时非线性？

**现阶段：不建议仅凭本次互补分析启动训练。**

支持“还有可能”的证据：互救非零，理想选择有有限改善空间。

不支持立刻开新实验的证据：两路判决高度相同；主要是边界附近的小幅翻转；幅度/变化速度不能一致指出非线性该负责的位置；MSE反而更差；没有不同时间尺度分工的直接证据；此前残差实验也未得到稳定增量。

同时，本次未分析新的储备池、未拟合选择器、未估计长时间记忆核，不能证明长/短新结构无效。若之后另行批准一次机制实验，应固定总节点、输入可见范围、读出直通与预算；同时对照线性、tanh、混合激活双支及同激活双时间尺度模型，区分收益来自激活还是时间尺度。不能削弱单分支、也不能把本轮验证当新盲测。此建议不是本轮训练授权。

## 9. 复现与文件

- `protocol.json`：分析前确定的对照、分层、指标、范围。
- `analyze.py`：只解码保存的验证预测，无fit调用；拒绝覆盖已完成summary。
- `four_cells.csv`：PAM6对、PAM4对、PAM4分量三种统计；`pair_summary.csv`：互救/重合/相关性。
- `metrics.json`：全部固定背景模型BER口径；`strata.csv`、`temporal_blocks_256.csv`：完整分层与时序，未只挑有利格。
- `paired_block_bootstrap.json`、`event_runs.json`、`observable_diagnostics.json`：区间、run和描述性AUC。
- `input_access_manifest.json`：正式成功运行读取的NPZ成员和标签区间、数组hash；事故另见第1节。
- `verification.json`：4项针对性检查、27组已有指标一致性、理想验证符号解码正确。没有训练或测试指标计算。

使用既有 `rc_baseline/.venv/bin/python`。若要重跑，创建 `rc_baseline/runs/` 下新目录，仅复制 `analyze.py`、`protocol.json` 和 `write_report.py` 后运行，保护本次结果：

```bash
mkdir rc_baseline/runs/validation_complementarity_replay
cp rc_baseline/runs/validation_complementarity_20261030/{analyze.py,protocol.json,write_report.py} rc_baseline/runs/validation_complementarity_replay/
rc_baseline/.venv/bin/python rc_baseline/runs/validation_complementarity_replay/analyze.py
rc_baseline/.venv/bin/python rc_baseline/runs/validation_complementarity_replay/write_report.py
```

全部新文件位于独立目录，旧模型和历史实验未修改。图表以英文标注避免跨机器字体缺失；解释及结论为中文。
'''
(p/'REPORT.md').write_text(doc)
print('REPORT.md written')
