"""波形级双分支可学习融合：原始实验文件不修改，全部注释使用中文。"""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,hashlib
import numpy as np
from fusion_layer import fusion_layer
from fusion_prefix_backend import branches,prefix_lms,prefix_cap,prefix_volterra
from receiver_metrics import channel_results,decoded_ber,BER_START,BER_TAIL,CAL_START,CAL_STOP
import rx_parallel_rc_lms as fixed
import rx_pbs_symbol_nltcp20250304offline as original

ROOT=Path(__file__).resolve().parent
NN_ROOT=fixed.NN_ROOT
DATANO=51200
TRAIN_RATIO=.3
BRANCH_FRACTION=2/3
MODES=("projection","weighted_sum")
SCHEMES=("projection","weighted_sum","fixed_weight","volterra_lms","pure_rc")
# 所有重算对照共用原Volterra基线的符号LMS配置，防止后端参数成为混杂因素。
SYMBOL_LMS=original.SYMBOL_LMS
TRAIN_SYMBOL=10000
WAVE_RC=tuple(dict(p,nodes=40) for p in fixed.WAVE_RC)
# 参考论文LIBS/NIRS独立分支，在此对应同一接收波形的线性/非线性RC独立分支。
# 参考论文拼接与注意力，在此对应实波形2→1投影，或复符号4→2投影及互补权重。
# 参考论文展平预测浓度，在此对应融合→CAP→LMS→NLTCP→PBS，不使用图像结构。
# 仅借鉴DFE的互补思想，没有判决反馈回路，不将此模型称为真正的DFE。

def load(name):
    return np.loadtxt(NN_ROOT/name).ravel()

def init_worker():
    global TARGET,WAVEFORM,REFERENCES
    TARGET=load("v1.txt")+1j*load("v2.txt")
    WAVEFORM=load("txdata/RF0.105_data36QAM.txt")
    REFERENCES=(load("decimal_PAM4_1.txt"),load("decimal_PAM4_2.txt"))

def metrics(outputs,test_start):
    # 原PBS函数和原BER区间保持不变，同时追加相同PBS权重下的无训练区测试BER。
    legacy=channel_results(outputs,REFERENCES)
    calibration=np.array([decoded_ber(x,REFERENCES,CAL_START,CAL_STOP) for x in outputs])
    weights=calibration[::-1]/calibration.sum() if calibration.sum() else np.array([.5,.5])
    combined=weights[0]*outputs[0]+weights[1]*outputs[1]
    strict=np.array([decoded_ber(x,REFERENCES,test_start,DATANO-BER_TAIL)
                     for x in (*outputs,combined)])
    return legacy,strict,weights.tolist()

def run_one_case(task):
    case,train_ratio,native,verbose=task
    train_symbols=int(DATANO*train_ratio)
    train_wave=4*train_symbols
    branch_end=int(train_wave*BRANCH_FRACTION)
    calibration=(branch_end+128,train_wave-512)
    test_start=max(BER_START,train_symbols+128)
    if TRAIN_SYMBOL+128 >= branch_end//4 or calibration[1]<=calibration[0]:
        raise ValueError("训练比例过小，无法容纳原LMS训练长度和独立融合校准区")
    outputs={name:[] for name in SCHEMES}
    records=[]
    for channel in (1,2):
        received=load(f"rxdata/rx{channel}_{case}.txt")
        if len(received)!=DATANO*4 or len(WAVEFORM)!=len(received):
            raise ValueError("波形必须完整包含204800点，防止时序错位")
        params=dict(WAVE_RC[channel-1])
        linear,nonlinear=branches(received,WAVEFORM,params,branch_end,native)
        if verbose:
            print(f"第{case}组 CH{1 if channel==1 else 3}：输入{received.shape}，线性分支{linear.shape}，非线性分支{nonlinear.shape}",flush=True)
        waves={"fixed_weight":fixed.w1*linear+fixed.w2*nonlinear,"pure_rc":nonlinear}
        # 两种融合模式均报告，不使用测试BER选择胜出模式。
        for mode in MODES:
            waves[mode],fitted=fusion_layer(linear,nonlinear,WAVEFORM,
                mode=mode,train_slice=calibration,verbose=verbose)
            records.append({"channel":1 if channel==1 else 3,"mode":mode,
                "coefficient":fitted.coefficient.tolist(),"bias":fitted.bias.tolist(),
                "fit_slice":list(calibration),"branch_train_end":branch_end})
        waves["volterra_lms"]=prefix_volterra(original.VOLTERRA[channel-1],received,WAVEFORM,branch_end)
        for name,wave in waves.items():
            symbols=prefix_cap(wave,4,DATANO,training_count=TRAIN_SYMBOL)
            recovered=prefix_lms(symbols,TARGET,SYMBOL_LMS[channel-1],TRAIN_SYMBOL,native)
            if len(symbols)!=DATANO or len(recovered)!=DATANO:
                raise ValueError("CAP/LMS输出必须与51200个NLTCP参考符号严格对齐")
            if verbose:
                print(f"第{case}组 CH{1 if channel==1 else 3} {name}：CAP{symbols.shape}→LMS{recovered.shape}",flush=True)
            outputs[name].append(recovered)
    results={}
    for name in SCHEMES:
        legacy,strict,weights=metrics(outputs[name],test_start)
        results[name]={"legacy":legacy.tolist(),"test":strict.tolist(),"pbs_weights":weights}
    return case,results,records

def plot_results(matrices,show=False):
    import matplotlib
    if not show: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 两种融合模式与三个公平重算基线在同一区间上比较。
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    labels={"projection":"Learned projection","weighted_sum":"Learned convex sum",
            "fixed_weight":"Fixed dual RC + LMS","volterra_lms":"Volterra-LMS + LMS",
            "pure_rc":"Single nonlinear RC + LMS"}
    for column,ax in enumerate(axes):
        for name in SCHEMES:
            ax.semilogy(range(1,20),np.maximum(matrices[name][:,column],np.finfo(float).tiny),
                        marker=".",label=labels[name])
        ax.set_title(("CH1","CH3","PBS")[column]);ax.set_xlabel("Experiment")
        ax.set_ylabel("Test BER");ax.grid(True);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(NN_ROOT/"dual_rc_fusion_comparison_test.png",dpi=180)
    if show: plt.show()
    plt.close(fig)

def run_all_experiments(train_ratio=TRAIN_RATIO,workers=4,native=False,show=False,verbose=True,fusion_mode="projection"):
    if fusion_mode not in MODES: raise ValueError("不支持的融合模式")
    if not 0<train_ratio<1: raise ValueError("train_ratio必须位于0和1之间")
    paths=("rx_parallel_rc_lms.py","linear_rc_equalizer.py","rclzl_w2s.py",
        "receiver_metrics.py","cap_demodulation.py","lms_1downs_testnan.py",
        "rx_pbs_symbol_nltcp20250304offline.py")
    protected={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    matrices={kind:{name:np.full((19,3),np.nan) for name in SCHEMES} for kind in ("legacy","test")}
    records=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for case,result,parameters in pool.map(run_one_case,
            [(n,train_ratio,native,verbose) for n in range(1,20)]):
            for name in SCHEMES:
                for kind in matrices:
                    matrices[kind][name][case-1]=result[name][kind]
                    np.savetxt(NN_ROOT/f"dual_rc_fusion_{name}_{kind}_berall.txt",
                        matrices[kind][name],fmt="%.16e")
            records.append({"case":case,"fusion":parameters,
                "pbs_weights":{name:result[name]["pbs_weights"] for name in SCHEMES}})
            print(f"第{case}组完成，投影融合测试BER={result['projection']['test']}",flush=True)
    summary={"primary_mode":fusion_mode,"train_ratio":train_ratio,"branch_train_fraction":BRANCH_FRACTION,
        "branch_wave_train_end":int(4*int(DATANO*train_ratio)*BRANCH_FRACTION),
        "symbol_train_samples":TRAIN_SYMBOL,"nodes":40,"wave_rc":WAVE_RC,
        "symbol_lms":SYMBOL_LMS,"fixed_weights":[fixed.w1,fixed.w2],
        "test_slice":[max(BER_START,int(DATANO*train_ratio)+128),DATANO-BER_TAIL],
        "legacy_slice":[BER_START,DATANO-BER_TAIL],
        "statistics_note":"旧区间保留复现，训练比例0.3下包含部分训练数据；主要比较使用test区间",
        "means":{},"parameters":records,"protected_sha256":protected,
        "reference_paper":"https://doi.org/10.1016/j.aca.2025.343899",
        "comparison_note":"全部重算对照采用相同训练区RMS、相同训练预算及相同符号LMS；旧文件未覆盖"}
    for name in SCHEMES:
        summary["means"][name]={k:matrices[k][name].mean(axis=0).tolist() for k in matrices}
        print(f"{name}：测试平均CH1/CH3/PBS BER={summary['means'][name]['test']}",flush=True)
    worst=max(summary["means"][n]["test"][2] for n in ("fixed_weight","volterra_lms"))
    summary["performance_criterion"]={mode:summary["means"][mode]["test"][2]<=worst for mode in MODES}
    for p,digest in protected.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("原实验文件意外改变："+p)
    (NN_ROOT/"dual_rc_fusion_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    # 主结果模式由运行参数预先指定，不依据测试指标挑选。
    np.savetxt(NN_ROOT/"dual_rc_fusion_berall.txt",matrices["legacy"][fusion_mode],fmt="%.16e")
    np.savetxt(NN_ROOT/"dual_rc_fusion_test_berall.txt",matrices["test"][fusion_mode],fmt="%.16e")
    plot_results(matrices["test"],show)
    return matrices,summary

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="RC线性/非线性波形双分支可学习融合")
    parser.add_argument("--fusion-mode",choices=MODES,default="projection")
    parser.add_argument("--train-ratio",type=float,default=TRAIN_RATIO)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--quiet",action="store_true")
    args=parser.parse_args()
    run_all_experiments(args.train_ratio,args.workers,args.native,args.show,not args.quiet,args.fusion_mode)
