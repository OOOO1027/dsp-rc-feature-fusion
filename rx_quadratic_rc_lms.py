"""单路tanh RC二次读出实验；公共CAP、符号LMS、NLTCP及PBS不修改。"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ[key]="1"
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,hashlib,time
import numpy as np
import rx_dual_rc_fusion_lms as common
from rx_soft_gate_rc_lms import validation_segments,robust_validation_score
from fusion_prefix_backend import prefix_cap,prefix_lms,prefix_volterra,fast_rc
from quadratic_rc_equalizer import (build_rc_features,expand_features,fit_nlms,
    fit_ridge_candidates,predict_readout,polynomial_weights)

ROOT=Path(__file__).resolve().parent
ALPHAS=(1e-6,1e-4,1e-2,1e-1)
SCHEMES=("linear_nlms","quadratic_nlms","linear_ridge","quadratic_ridge","volterra_lms")
PRIMARY="quadratic_ridge"

def init_worker(): common.init_worker()

def backend(wave,channel,native=False):
    symbols=prefix_cap(wave,4,len(wave)//4,training_count=common.TRAIN_SYMBOL)
    return prefix_lms(symbols,common.TARGET[:len(symbols)],common.SYMBOL_LMS[channel],common.TRAIN_SYMBOL,native)

def save_model(case,channel,name,model,base_width,normalization,params):
    w1,w2,direct,bias=polynomial_weights(model,base_width)
    folder=common.NN_ROOT/"quadratic_rc_models"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f"case{case:02d}_CH{channel}_{name}.npz",
        coefficient=model.coefficient,bias=model.bias,target_scale=model.target_scale,
        degree=model.degree,nodes=model.nodes,alpha=model.alpha,solver=model.solver,
        W1=w1,W2=w2,direct=direct,physical_bias=bias,
        normalization_json=json.dumps(normalization),params_json=json.dumps(params))

def run_case(task):
    case,ratio=task
    train_symbols=int(51200*ratio)
    branch_end=int(4*train_symbols*common.BRANCH_FRACTION)
    start=branch_end//4+128;stop=train_symbols-128
    if branch_end//4<=common.TRAIN_SYMBOL+128: raise ValueError("训练比例不足以保留公共链路训练区")
    segments=validation_segments(start,stop)
    references=tuple(x[:train_symbols] for x in common.REFERENCES)
    outputs={name:[] for name in SCHEMES}
    records=[]
    started=time.perf_counter()
    for channel in range(2):
        params=dict(common.WAVE_RC[channel],nodes=40,seed=100)
        received=common.load(f"rxdata/rx{channel+1}_{case}.txt")
        if len(received)!=204800: raise ValueError("接收波形必须为204800点")
        base,scale,fit_slice,normalization=build_rc_features(received,common.WAVEFORM,params,branch_end)
        desired=common.WAVEFORM[fit_slice]/scale
        channel_records={"channel":1 if channel==0 else 3,"params":params,
                         "normalization":normalization,"models":{}}
        for degree,label in ((1,"linear"),(2,"quadratic")):
            training=expand_features(base[fit_slice],params["nodes"],degree)
            print(f"第{case}组 CH{channel_records['channel']} {label}：状态(204800,40)，训练特征{training.shape}",flush=True)
            # 同一矩阵先训练原NLMS，再原地标准化给岭回归；不分配全长二次矩阵。
            model=fit_nlms(training,desired,nodes=40,degree=degree,target_scale=scale,
                           step=params["lms_step"],epochs=params["lms_epochs"])
            name=label+"_nlms"
            wave=predict_readout(base,model)
            recovered=backend(wave,channel)
            outputs[name].append(recovered)
            if case==1 and degree==1:
                original=fast_rc(received,common.WAVEFORM,params,branch_end).real
                np.testing.assert_allclose(wave,original,rtol=1e-9,atol=1e-9)
                other=backend(wave,channel,True)
                np.testing.assert_allclose(recovered,other,rtol=1e-9,atol=1e-9)
            save_model(case,channel_records["channel"],name,model,base.shape[1],normalization,params)
            channel_records["models"][name]={"feature_count":training.shape[1],"solver":"nlms",
                "epochs":params["lms_epochs"],"step":params["lms_step"]}
            candidates=fit_ridge_candidates(training,desired,nodes=40,degree=degree,
                                             target_scale=scale,alphas=ALPHAS)
            del training
            history=[];best=float("inf");selected=None
            for candidate in candidates:
                prefix=predict_readout(base[:4*train_symbols],candidate)
                validation=backend(prefix,channel)
                score,parts=robust_validation_score(validation,references,segments)
                history.append({"alpha":candidate.alpha,"validation_score":score,"segment_bers":parts})
                if score<best:
                    best=score;selected=candidate
            name=label+"_ridge"
            wave=predict_readout(base,selected)
            outputs[name].append(backend(wave,channel))
            save_model(case,channel_records["channel"],name,selected,base.shape[1],normalization,params)
            channel_records["models"][name]={"feature_count":len(selected.coefficient),"solver":"ridge",
                "selected_alpha":selected.alpha,"validation_segments":segments,"search":history}
            print(f"第{case}组 CH{channel_records['channel']} {name}：alpha={selected.alpha}，验证评分={best:.8g}",flush=True)
        volterra=prefix_volterra(common.original.VOLTERRA[channel],received,common.WAVEFORM,branch_end)
        outputs["volterra_lms"].append(backend(volterra,channel))
        records.append(channel_records)
        print(f"第{case}组 CH{channel_records['channel']}：全部波形(204800,)→CAP/LMS(51200,)→NLTCP/PBS",flush=True)
    results={}
    for name in SCHEMES:
        legacy,test,weights=common.metrics(outputs[name],max(common.BER_START,train_symbols+128))
        results[name]={"legacy":legacy.tolist(),"test":test.tolist(),"pbs_weights":weights}
    return case,results,records,time.perf_counter()-started

def plot(matrices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    for column,ax in enumerate(axes):
        for name in SCHEMES:
            ax.semilogy(range(1,20),np.maximum(matrices[name][:,column],1e-10),marker=".",label=name)
        ax.set_title(("CH1","CH3","PBS")[column]);ax.set_xlabel("Experiment")
        ax.set_ylabel("Test BER");ax.grid(True);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(common.NN_ROOT/"quadratic_rc_comparison_test.png",dpi=180);plt.close(fig)

def run_all(train_ratio=.3,workers=2):
    if not 0<train_ratio<1: raise ValueError("训练比例必须位于0和1之间")
    paths=("fusion_prefix_backend.py","rclzl_w2s.py","receiver_metrics.py",
           "rx_dual_rc_fusion_lms.py","rx_soft_gate_rc_lms.py","gating_fusion_layer.py",
           "rx_pbs_symbol_nltcp20250304offline.py")
    protected={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    matrices={kind:{name:np.full((19,3),np.nan) for name in SCHEMES} for kind in ("legacy","test")}
    records=[]
    began=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for case,result,parameters,elapsed in pool.map(run_case,[(i,train_ratio) for i in range(1,20)]):
            for kind in matrices:
                for name in SCHEMES:
                    matrices[kind][name][case-1]=result[name][kind]
                    np.savetxt(common.NN_ROOT/f"quadratic_rc_{name}_{kind}_berall.txt",matrices[kind][name],fmt="%.16e")
            records.append({"case":case,"channels":parameters,"seconds":elapsed,
                            "pbs_weights":{n:result[n]["pbs_weights"] for n in SCHEMES}})
            print(f"第{case}组完成，二次岭读出测试BER={result[PRIMARY]['test']}，耗时{elapsed:.1f}s",flush=True)
    verification={"full_length_waveform":204800,"full_length_symbols":51200,
                  "case1_original_NLMS_waveform_and_native_symbol_LMS_parity":True}
    for kind in matrices:
        for name in SCHEMES:
            assert matrices[kind][name].shape==(19,3) and np.isfinite(matrices[kind][name]).all()
    # 核对公共Volterra基线逐项未变化。
    prior=common.NN_ROOT/"volterra_nonlinearity_summary.json"
    if prior.exists():
        old=json.loads(prior.read_text(encoding="utf-8"))
        if old["train_ratio"]==train_ratio:
            for kind in matrices:
                np.testing.assert_array_equal(matrices[kind]["volterra_lms"],
                    np.loadtxt(common.NN_ROOT/f"volterra_nonlinearity_volterra_{kind}_berall.txt"))
            verification["all19_existing_volterra_exact_match"]=True
    means={name:{kind:matrices[kind][name].mean(axis=0).tolist() for kind in matrices} for name in SCHEMES}
    for name in SCHEMES: print(f"{name}平均测试CH1/CH3/PBS BER={means[name]['test']}",flush=True)
    for name,h in protected.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    verification.update(all_19_matrices_finite=True,protected_files_unchanged=True)
    summary={"primary":PRIMARY,"train_ratio":train_ratio,"nodes":40,"seed":100,"wave_rc":common.WAVE_RC,
        "branch_train_end":int(4*int(51200*train_ratio)*2/3),"symbol_lms":common.SYMBOL_LMS,
        "test_slice":[max(10128,int(51200*train_ratio)+128),51072],"alphas":ALPHAS,
        "ridge_objective":"mean_squared_training_error_plus_alpha_times_standardized_slope_norm_squared",
        "ridge_selection":"four_training_validation_segments_mean_BER_plus_0.25_std",
        "quadratic_state_features":820,"readout_feature_counts":[911,927],
        "direct_input_features":"保留原始I/Q滑窗直通；只对40个状态增加二次项",
        "parameters":records,"means":means,"verification":verification,"protected_sha256":protected,
        "elapsed_seconds":time.perf_counter()-began,
        "interpretation":"主模型预先指定为二次岭读出；同时报告原NLMS及匹配线性岭对照，不按测试BER选求解器"}
    (common.NN_ROOT/"quadratic_rc_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    np.savetxt(common.NN_ROOT/"quadratic_rc_test_berall.txt",matrices["test"][PRIMARY],fmt="%.16e")
    np.savetxt(common.NN_ROOT/"quadratic_rc_berall.txt",matrices["legacy"][PRIMARY],fmt="%.16e")
    plot(matrices["test"])
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="40节点RC显式二次读出与匹配对照")
    parser.add_argument("--train-ratio",type=float,default=.3)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    run_all(args.train_ratio,args.workers)
