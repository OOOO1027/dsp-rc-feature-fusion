"""双RC独立分支的波形域软门控，公共CAP/LMS/NLTCP/PBS完全复用。"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ[key]="1"
import argparse,json,hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import rx_dual_rc_fusion_lms as common
from fusion_prefix_backend import fast_linear,fast_rc,prefix_cap,prefix_lms,prefix_volterra
from linear_rc_equalizer import _linear_rc_equalize
from rclzl_w2s import rc_wave_equalize
from receiver_metrics import NLTCP_demodulation,BITS_PER_SYMBOL
from gating_fusion_layer import fit_gating,gating_fusion_layer

ROOT=Path(__file__).resolve().parent
SCHEMES=("soft_gate","fixed_weight","pure_rc","volterra_lms")


# 同一波形的两个时间尺度，各自独立拟合；仅修改窗口长度。
LINEAR_TAPS=13
NONLINEAR_TAPS=41
ROBUSTNESS_PENALTY=.25

def multiscale_branches(wave,target,settings,train_end,native=False):
    linear_settings=dict(settings,taps=LINEAR_TAPS)
    nonlinear_settings=dict(settings,taps=NONLINEAR_TAPS)
    if native:
        linear=_linear_rc_equalize(wave,target,train_samples=train_end,**linear_settings)[0].real
        nonlinear=rc_wave_equalize(wave,target,train_samples=train_end,**nonlinear_settings)[0]
    else:
        linear=fast_linear(wave,target,linear_settings,train_end).real
        nonlinear=fast_rc(wave,target,nonlinear_settings,train_end).real
    return linear,nonlinear

def validation_segments(start,stop):
    # 固定训练预算内划分四段，不把重复统计伪称为新增独立样本。
    edges=np.linspace(start,stop,5,dtype=int)
    if np.min(np.diff(edges))<256: raise ValueError("验证区不足以划分四个稳定片段")
    return [(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:])]

def robust_validation_score(recovered,references,segments):
    # 离散判决只算一次，各段仍沿用原NLTCP和三比特BER定义。
    decoded=[NLTCP_demodulation(len(recovered),part)[1] for part in (recovered.real,recovered.imag)]
    values=np.array([sum(common.original.biterr(d[a:b],r[a:b],BITS_PER_SYMBOL)
                            for d,r in zip(decoded,references))/2 for a,b in segments])
    return float(values.mean()+ROBUSTNESS_PENALTY*values.std()),values.tolist()

def init_worker(): common.init_worker()

def backend(wave,channel,native=False):
    count=len(wave)//4
    symbols=prefix_cap(wave,4,count,training_count=common.TRAIN_SYMBOL)
    return prefix_lms(symbols,common.TARGET[:count],common.SYMBOL_LMS[channel],common.TRAIN_SYMBOL,native)

def run_case(task):
    case,ratio,native=task
    train_symbols=int(common.DATANO*ratio)
    train_end=4*train_symbols
    branch_end=int(train_end*common.BRANCH_FRACTION)
    validation_start=branch_end//4+128
    validation_stop=train_symbols-128
    if validation_start>=validation_stop or branch_end//4<=common.TRAIN_SYMBOL+128:
        raise ValueError("训练比例不足以容纳分支、符号LMS和独立门控验证区")
    outputs={name:[] for name in SCHEMES}
    records=[]
    for channel in range(2):
        wave=common.load(f"rxdata/rx{channel+1}_{case}.txt")
        if len(wave)!=204800: raise ValueError("接收波形需要204800点")
        settings=dict(common.WAVE_RC[channel],nodes=40,seed=100)
        # 两路只接收同一原始波形，分支仅拟合前20%，后10%留给门控选择。
        linear,nonlinear=multiscale_branches(wave,common.WAVEFORM,settings,branch_end,native)
        count=min(len(linear),len(nonlinear),len(wave))
        if count<train_end: raise ValueError("分支对齐后不足训练区，禁止补造标签")
        linear,nonlinear,wave=linear[:count],nonlinear[:count],wave[:count]
        if count!=204800: raise ValueError("实验CAP链路要求完整204800点，分支截断后长度不足")
        # 候选评分只接收训练前缀，验证标签不包含任何测试符号。
        references=tuple(x[:train_symbols] for x in common.REFERENCES)
        segments=validation_segments(validation_start,validation_stop)
        score_details=[]
        def score(candidate):
            recovered=backend(candidate,channel,native)
            value,parts=robust_validation_score(recovered,references,segments)
            score_details.append({"score":value,"segment_bers":parts})
            return value
        params,history=fit_gating(linear[:train_end],nonlinear[:train_end],wave[:train_end],
            taps=(LINEAR_TAPS,NONLINEAR_TAPS),fit_end=branch_end,score=score,refine_constant=True)
        for item,detail in zip(history,score_details):
            item["validation_score"]=item.pop("validation_ber")
            item.update(detail)
        fused,gates,_=gating_fusion_layer(linear,nonlinear,wave,wave,parameters=params)
        statistics={"mean":gates.mean(axis=0).tolist(),"min":gates.min(axis=0).tolist(),
                    "max":gates.max(axis=0).tolist(),"std":gates.std(axis=0).tolist()}
        print(f"第{case}组 CH{1 if channel==0 else 3}：输入{wave.shape}，分支{linear.shape}/{nonlinear.shape}，融合{fused.shape}；门控{statistics}",flush=True)
        records.append({"channel":1 if channel==0 else 3,"weight":params.weight.tolist(),
            "bias":params.bias.tolist(),"mean":params.mean.tolist(),"scale":params.scale.tolist(),
            "taps":params.taps,"gate_statistics":statistics,"search":history,
            "validation_slice":[validation_start,validation_stop],"validation_segments":segments,
            "constant_retry":any(h["stage"]==3 for h in history),
            "feature_scale_correlation":float(np.corrcoef(
                gating_fusion_layer(linear[:branch_end],nonlinear[:branch_end],wave[:branch_end],wave[:branch_end],
                    parameters=params,return_features=True)[3].T)[1,4])})
        waves={"soft_gate":fused,"fixed_weight":common.fixed.w1*linear+common.fixed.w2*nonlinear,
               "pure_rc":nonlinear,"volterra_lms":prefix_volterra(common.original.VOLTERRA[channel],wave,common.WAVEFORM,branch_end)}
        for name,value in waves.items():
            recovered=backend(value,channel,native)
            if len(recovered)!=51200: raise ValueError("CAP/LMS输出必须为51200个复符号")
            outputs[name].append(recovered)
        print(f"第{case}组公共链路：CAP/LMS(51200,)→NLTCP→PBS",flush=True)
    result={}
    for name in SCHEMES:
        legacy,test,weights=common.metrics(outputs[name],max(common.BER_START,train_symbols+128))
        result[name]={"legacy":legacy.tolist(),"test":test.tolist(),"pbs_weights":weights}
    return case,result,records

def plot(matrices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for column,ax in enumerate(axes):
        for name in SCHEMES:
            ax.semilogy(range(1,20),np.maximum(matrices[name][:,column],1e-10),marker=".",label=name)
        ax.set_title(("CH1","CH3","PBS")[column]);ax.set_xlabel("Experiment")
        ax.set_ylabel("Test BER");ax.grid(True);ax.legend()
    fig.tight_layout();fig.savefig(common.NN_ROOT/"soft_gate_rc_comparison_test.png",dpi=180);plt.close(fig)

def run_all(train_ratio=.3,workers=4,native=False):
    if not 0<train_ratio<1: raise ValueError("训练比例必须位于0与1之间")
    protected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/x for x in ("rx_dual_rc_fusion_lms.py","fusion_prefix_backend.py",
                         "receiver_metrics.py","linear_rc_equalizer.py","rclzl_w2s.py")]}
    matrices={kind:{name:np.full((19,3),np.nan) for name in SCHEMES} for kind in ("legacy","test")}
    records=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for case,result,parameters in pool.map(run_case,[(i,train_ratio,native) for i in range(1,20)]):
            records.append({"case":case,"gates":parameters,"pbs_weights":{n:result[n]["pbs_weights"] for n in SCHEMES}})
            for kind in matrices:
                for name in SCHEMES:
                    matrices[kind][name][case-1]=result[name][kind]
                    np.savetxt(common.NN_ROOT/f"soft_gate_rc_{name}_{kind}_berall.txt",matrices[kind][name],fmt="%.16e")
            print(f"第{case}组测试BER：{result['soft_gate']['test']}",flush=True)
    means={name:{kind:matrices[kind][name].mean(axis=0).tolist() for kind in matrices} for name in SCHEMES}
    for name in SCHEMES: print(f"{name}平均测试CH1/CH3/PBS BER={means[name]['test']}",flush=True)
    for name,digest in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    channels=[g for r in records for g in r["gates"]]
    diagnostics={"dynamic_std_threshold":1e-6,
        "dynamic_channels":sum(max(g["gate_statistics"]["std"])>1e-6 for g in channels),
        "clear_dynamic_std_threshold":.01,
        "clear_dynamic_channels":sum(max(g["gate_statistics"]["std"])>.01 for g in channels),
        "constant_channels":sum(max(g["gate_statistics"]["std"])<=1e-6 for g in channels),
        "retry_channels":sum(g["constant_retry"] for g in channels),
        "rescued_channels":sum(g["constant_retry"] and max(g["gate_statistics"]["std"])>1e-6 for g in channels),
        "short_long_mean_feature_correlation":float(np.mean([g["feature_scale_correlation"] for g in channels]))}
    print(f"门控汇总：{diagnostics}",flush=True)
    summary={"experiment_version":"multiscale_robust_v2","gating_diagnostics":diagnostics,
        "branch_configs":[{"linear":dict(p,taps=LINEAR_TAPS),"nonlinear":dict(p,taps=NONLINEAR_TAPS)} for p in common.WAVE_RC],
        "train_ratio":train_ratio,"wave_rc":common.WAVE_RC,"symbol_lms":common.SYMBOL_LMS,
        "seed":100,"nodes":40,"branch_train_end":int(4*int(51200*train_ratio)*2/3),
        "test_slice":[max(common.BER_START,int(51200*train_ratio)+128),51072],
        "linear_taps":LINEAR_TAPS,"nonlinear_taps":NONLINEAR_TAPS,
        "validation_method":"four_segments_mean_plus_0.25_std","robustness_penalty":ROBUSTNESS_PENALTY,
        "constant_retry":"fine_coordinates_and_cross_scale_pairs_with_bias_restarts",
        "training_method":"robust_validation_BER_grid_then_conditional_second_search_no_MSE",
        "fallback_reason":"离散NLTCP和自适应LMS未建立稳定可微反传，使用用户允许的验证BER候选搜索；不声称实现梯度微调",
        "parameters":records,"means":means,"protected_sha256":protected,
        "criterion_met":means["soft_gate"]["test"][2]<=max(means[n]["test"][2] for n in ("fixed_weight","pure_rc"))}
    (common.NN_ROOT/"soft_gate_rc_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savetxt(common.NN_ROOT/"soft_gate_rc_test_berall.txt",matrices["test"]["soft_gate"],fmt="%.16e")
    np.savetxt(common.NN_ROOT/"soft_gate_rc_berall.txt",matrices["legacy"]["soft_gate"],fmt="%.16e")
    plot(matrices["test"])
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="双分支逐采样点软门控均衡")
    parser.add_argument("--train-ratio",type=float,default=.3)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--native",action="store_true")
    args=parser.parse_args()
    run_all(args.train_ratio,args.workers,args.native)
