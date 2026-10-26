"""时间尺度互补双RC各自二次读出的独立实验，公共DSP及PBS权重冻结。"""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
from pathlib import Path
import json,hashlib,argparse,time
import numpy as np
from multiscale_runtime import threadpool_limits,THREADPOOL_AVAILABLE
import multiscale_runtime as public
from ch3_polynomial_readout import fit_models,predict_models,save_candidates,load_candidates
from multiscale_quadratic_rc import build_features,fit_projection,fusion_layer,convex_projection

ROOT=public.ROOT;NN=public.NN
CACHE=NN/"multiscale_quadratic_models"
LOCK=NN/"multiscale_quadratic_selection.json"
TRAIN_END=40960;PREFIX_WAVE=61440
ALPHAS=(1e-5,1e-4,.001,.0031622776601683794,.01)
BRANCHES=("short_linear","long_tanh")
SEARCH=public.SEARCH;CONFIRM=public.CONFIRM
CONVEX=tuple(float(x) for x in np.linspace(0,1,11))
CASES=tuple(range(1,20))
BACKEND=public.backend
COMMON=public.COMMON

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def parameters(ch,branch):
    # 短递归稳定且快，长递归泄露较小；同一随机seed不代表两种状态相同。
    common=dict(nodes=40,seed=100,passthrough_taps=25 if ch==0 else 97)
    if branch=="short_linear":return dict(common,taps=13,activation="identity",gamma=1.,spectral_radius=.5)
    if branch=="long_tanh":return dict(common,taps=41,activation="tanh",gamma=.2,spectral_radius=.95)
    raise ValueError("未知分支")

def model_path(case,ch,branch):return CACHE/f"case{case:02d}_CH{1 if ch==0 else 3}_{branch}.npz"
def wave_path(case,ch,branch):return CACHE/f"case{case:02d}_CH{1 if ch==0 else 3}_{branch}_waves.npz"

def prefix_wave(case,ch):return COMMON.load(f"rxdata/rx{ch+1}_{case}.txt")[:PREFIX_WAVE]

def stats(symbols,segments):
    return public.rates(public.channel_counts(symbols,COMMON.REFERENCES,segments),segments).tolist()

def train_branch(case,ch,branch,alphas=ALPHAS):
    path=model_path(case,ch,branch);wp=wave_path(case,ch,branch)
    if path.exists() and wp.exists():
        models,params,norm,history,info=load_candidates(path)
        if all(any(m.alpha==a for m in models) for a in alphas):return models,params,norm,history,info
    params=parameters(ch,branch)
    wave=prefix_wave(case,ch)
    features,scale,fit,norm=build_features(wave,COMMON.WAVEFORM[:PREFIX_WAVE],params,TRAIN_END)
    models,info=fit_models(features[fit],COMMON.WAVEFORM[fit]/scale,nodes=40,degree=2,target_scale=scale,alphas=alphas,verbose=False)
    waves=predict_models(features,models)
    history=[]
    for i,m in enumerate(models):
        symbols=BACKEND(waves[:,i],ch)
        history.append(dict(alpha=m.alpha,fit_bers=stats(symbols,public.FIT)))
    save_candidates(path,models,params,norm,history,info)
    np.savez_compressed(wp,alphas=np.array([m.alpha for m in models]),waves=waves)
    print(f"分支拟合：组{case} CH{1 if ch==0 else 3} {branch} 输入{wave.shape} 状态({len(wave)},40) 二次项820 读出维度{info['features']} 输出{waves.shape}",flush=True)
    return models,params,norm,history,info

def selected_wave(case,ch,branch,alpha):
    train_branch(case,ch,branch,(alpha,))
    with np.load(wave_path(case,ch,branch)) as data:
        index=int(np.where(data["alphas"]==alpha)[0][0]);return data["waves"][:,index]

def select_alphas():
    result={}
    for ch in range(2):
        result[str(ch)]={}
        for branch in BRANCHES:
            histories=[train_branch(case,ch,branch)[3] for case in SEARCH]
            rows=[]
            for alpha in ALPHAS:
                bers=[next(x["fit_bers"] for x in h if x["alpha"]==alpha) for h in histories]
                rows.append(dict(alpha=alpha,score=float(public.robust(bers).mean())))
            result[str(ch)][branch]=min(rows,key=lambda x:x["score"])["alpha"]
            print(f"校准冻结岭强度 CH{1 if ch==0 else 3} {branch}：{result[str(ch)][branch]}",flush=True)
            save(CACHE/f"CH{ch}_{branch}_alpha_search.json",dict(grid=rows,selected=result[str(ch)][branch]))
    return result

def projection_names():return ("trained_projection",)+tuple(f"convex_{w:.1f}" for w in CONVEX)

def get_projection(name,trained):
    if name=="trained_projection":return trained
    if name.startswith("convex_"):return convex_projection(float(name.split("_")[1]))
    if name=="short_only":return convex_projection(1.)
    if name=="long_only":return convex_projection(0.)
    raise ValueError("未知融合方式")

def prepare_projections(alphas):
    records={str(ch):{} for ch in range(2)}
    for case in CASES:
        for ch in range(2):
            waves=[selected_wave(case,ch,branch,alphas[str(ch)][branch]) for branch in BRANCHES]
            margin=max(41,parameters(ch,BRANCHES[0])["passthrough_taps"])//2
            trained=fit_projection(*waves,COMMON.WAVEFORM[:PREFIX_WAVE],slice(margin+1,TRAIN_END-margin))
            modes={}
            for name in projection_names():
                symbols=BACKEND(fusion_layer(*waves,get_projection(name,trained)),ch)
                modes[name]=dict(fit=stats(symbols,public.FIT),check=stats(symbols,public.CHECK))
            records[str(ch)][str(case)]=dict(trained=trained,modes=modes,
                branch_output_correlation=float(np.corrcoef(waves[0][40960:61440],waves[1][40960:61440])[0,1]))
        print(f"融合校准维度：组{case} 两分支各61440 → 波形投影61440 → CAP/LMS15360",flush=True)
    save(CACHE/"projection_validation.json",records)
    return records

def choose_projections(records):
    selected={};confirmation={}
    for ch in range(2):
        table=records[str(ch)]
        scores={name:float(public.robust([table[str(case)]["modes"][name]["fit"] for case in SEARCH]).mean()) for name in projection_names()}
        chosen=min(scores,key=scores.get)
        # 对照分支按校准段固定选择，不能在确认区逐组挑最优分支。
        short="convex_1.0";long="convex_0.0"
        reference=min((short,long),key=lambda name:scores[name])
        trial=[table[str(case)]["modes"][chosen]["check"] for case in CONFIRM]
        control=[table[str(case)]["modes"][reference]["check"] for case in CONFIRM]
        acceptance=public.accept(trial,control)
        selected[str(ch)]=chosen
        confirmation[str(ch)]=dict(reference=reference,acceptance=acceptance,fit_scores=scores,
            note="未通过时仍输出独立实验结果，不替换原最佳模型，不根据测试结果重选融合")
        print(f"CH{1 if ch==0 else 3}融合锁定{chosen}，对照{reference}，确认{acceptance}",flush=True)
    return selected,confirmation

def impulse_diagnostic():
    # 固定合成脉冲只测递归响应，不读取实验测试标签，也不据此改参数。
    wave=np.zeros(2400);wave[1200]=1.
    norm=dict(input_rms=1.,input_window_scale=1.,target_rms=1.)
    result={}
    for branch in BRANCHES:
        params=parameters(0,branch)
        states=build_features(wave,None,params,1000,norm)[0][:,:40]
        zero=build_features(np.zeros_like(wave),None,params,1000,norm)[0][:,:40]
        energy=np.sum((states-zero)**2,axis=1)
        cumulative=np.cumsum(energy)/energy.sum()
        a=int(np.searchsorted(cumulative,.05));b=int(np.searchsorted(cumulative,.95))
        result[branch]=dict(energy90_width=b-a+1,start_relative_to_impulse=a-1200,stop_relative_to_impulse=b-1200)
    return result

def manifest():
    CACHE.mkdir(exist_ok=True)
    names=("fusion_prefix_backend.py","receiver_metrics.py","quadratic_rc_equalizer.py","ch3_polynomial_readout.py",
        "pbs_validation_diagnostic.py","memory_passthrough_sweep.py")
    protected={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    files=(public.LOCK,NN/"pbs_validation_RC_calibrated_test_berall.txt",NN/"pbs_validation_Volterra_calibrated_test_berall.txt")
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    config=dict(parameters={str(ch):{branch:parameters(ch,branch) for branch in BRANCHES} for ch in range(2)},
        alphas=ALPHAS,convex=CONVEX,fit=public.FIT,check=public.CHECK,search=SEARCH,confirm=CONFIRM,protected=protected,inputs=inputs,
        module_hash=hashlib.sha256((ROOT/"multiscale_quadratic_rc.py").read_bytes()).hexdigest(),
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_hash=hashlib.sha256((ROOT/"multiscale_runtime.py").read_bytes()).hexdigest())
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    path=CACHE/"manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["fingerprint"]!=fingerprint:raise ValueError("时间尺度缓存与配置不符")
    save(path,dict(fingerprint=fingerprint,configuration=config))
    return protected,fingerprint

def run():
    COMMON.init_worker();protected,fingerprint=manifest();began=time.perf_counter()
    previous=json.loads(public.LOCK.read_text(encoding="utf-8"))
    pbs=[previous["validation"]["RC"][str(case)]["weights"][previous["decisions"]["RC"]["selected"]] for case in CASES]
    if not np.all(np.asarray(pbs)==.59):raise ValueError("当前实验要求既有RC PBS锁定为0.59，禁止隐式换策略")
    alphas=select_alphas();records=prepare_projections(alphas)
    projections,confirmation=choose_projections(records)
    selected=dict(fingerprint=fingerprint,alphas=alphas,projections=projections,confirmation=confirmation,
        parameters={str(ch):{branch:parameters(ch,branch) for branch in BRANCHES} for ch in range(2)},
        per_case_trained_projection={ch:{case:item["trained"] for case,item in table.items()} for ch,table in records.items()},
        validation_records=records,pbs_CH1_weight=.59,impulse_diagnostic=impulse_diagnostic(),
        protected=protected,test_used_for_selection=False,fit_end=TRAIN_END,total_training_prefix=PREFIX_WAVE,
        test_slice=[15488,51072],warning="两分支各40节点，总容量较单分支增加；不同窗口不等同于统计独立输入",
        seconds_before_test=time.perf_counter()-began)
    public.check_protection(protected)
    save(LOCK,selected)
    print("双分支二次读出及融合已锁定，开始最终十九组测试，PBS保持0.59/0.41",flush=True)
    result=evaluate(json.loads(LOCK.read_text(encoding="utf-8")))
    result["elapsed_seconds"]=time.perf_counter()-began
    save(NN/"multiscale_quadratic_summary.json",result)
    return result

def frozen_branch(case,ch,branch,alpha,wave):
    models,params,norm,_,_=load_candidates(model_path(case,ch,branch))
    model=next(x for x in models if x.alpha==alpha)
    features,_,_,_=build_features(wave,None,params,TRAIN_END,norm)
    return predict_models(features,[model])[:,0]

def evaluate(selected):
    if json.loads(LOCK.read_text(encoding="utf-8"))!=selected:raise ValueError("必须使用已冻结配置")
    COMMON.init_worker()
    names=("fused","short_only","long_only")
    matrix={kind:{name:np.empty((19,3)) for name in names} for kind in ("legacy","test")}
    shapes=[]
    for case in CASES:
        outputs={name:[] for name in names}
        for ch in range(2):
            wave=COMMON.load(f"rxdata/rx{ch+1}_{case}.txt")
            if len(wave)!=204800:raise ValueError("接收波形必须为204800点")
            branches=[frozen_branch(case,ch,branch,selected["alphas"][str(ch)][branch],wave) for branch in BRANCHES]
            trained=selected["per_case_trained_projection"][str(ch)][str(case)]
            mode=selected["projections"][str(ch)]
            for name in names:
                projection=get_projection(mode if name=="fused" else name,trained)
                fused=fusion_layer(*branches,projection)
                if len(fused)!=204800:raise ValueError("融合波形长度不足204800，禁止补造样本")
                symbols=BACKEND(fused,ch)
                if len(symbols)!=51200:raise ValueError("CAP/LMS后须为51200符号")
                outputs[name].append(symbols)
            cached=selected["validation_records"][str(ch)][str(case)]["modes"][mode]["check"]
            np.testing.assert_array_equal(np.rint(np.array(stats(outputs["fused"][-1],public.CHECK))*7296),
                np.rint(np.array(cached)*7296))
            shapes.append(dict(case=case,channel=1 if ch==0 else 3,branch_wave=204800,fused_wave=204800,symbols=51200))
        for kind,segment in (("legacy",((10128,51072),)),("test",((15488,51072),))):
            for name,pair in outputs.items():
                individual=[public.rates(public.channel_counts(x,COMMON.REFERENCES,segment),segment)[0] for x in pair]
                pbs=public.rates(public.weighted_counts(*pair,np.array([selected["pbs_CH1_weight"]]),segment,COMMON.REFERENCES),segment)[0,0]
                matrix[kind][name][case-1]=[*individual,pbs]
        print(f"冻结测试{case}/19 融合CH1/CH3/PBS={matrix['test']['fused'][case-1].tolist()}",flush=True)
    for kind in matrix:
        for name,data in matrix[kind].items():
            assert data.shape==(19,3) and np.isfinite(data).all()
            np.savetxt(NN/f"multiscale_quadratic_{name}_{kind}_berall.txt",data,fmt="%.16e")
    baseline=np.loadtxt(NN/"pbs_validation_RC_calibrated_test_berall.txt")
    stats_result={name:public.paired_statistics(data[:,2]-baseline[:,2]) for name,data in matrix["test"].items()}
    result=dict(selection=selected,means={name:{kind:matrix[kind][name].mean(axis=0).tolist() for kind in matrix} for name in names},
        paired_PBS_difference_vs_previous=stats_result,shapes=shapes,
        verification=dict(all_matrices_finite=True,prefix_full_validation_error_counts_exact=True,test_used_for_selection=False))
    public.check_protection(selected["protected"]);result["verification"]["protected_files_unchanged"]=True
    save(NN/"multiscale_quadratic_summary.json",result)
    plot(matrix["test"])
    for name in names:print(f"{name}平均CH1/CH3/PBS={result['means'][name]['test']}",flush=True)
    return result

def reference_ber_figure(berall):
    """复现原接收脚本的单面板、三条曲线及log10(BER)纵轴。"""
    import matplotlib.pyplot as plt
    values=np.asarray(berall,dtype=float)
    if values.shape!=(19,3) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError("BER矩阵须为非负有限19×3数组")
    fig,ax=plt.subplots(figsize=(8,5))
    # 零BER保留为负无穷，不将它伪造为某个非零误码率。
    with np.errstate(divide="ignore"):ax.plot(np.arange(1,20),np.log10(values))
    ax.legend(["CH1","CH3","PBS"])
    ax.set_xlabel("experiment index")
    ax.set_ylabel("log10(BER)")
    ax.set_title("BER evolution over experiments")
    ax.grid(True)
    fig.tight_layout()
    return fig

def save_reference_ber_plot(berall,path):
    import matplotlib.pyplot as plt
    fig=reference_ber_figure(berall)
    fig.savefig(path,dpi=200)
    plt.close(fig)
    print(f"已保存原脚本格式BER图：{path}",flush=True)

def plot(matrix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    save_reference_ber_plot(matrix["fused"],NN/"multiscale_quadratic_ber_test.png")
    legacy=NN/"multiscale_quadratic_fused_legacy_berall.txt"
    if legacy.exists():save_reference_ber_plot(np.loadtxt(legacy),NN/"multiscale_quadratic_ber_legacy.png")
    series=dict(matrix,previous_single=np.loadtxt(NN/"pbs_validation_RC_calibrated_test_berall.txt"),
        calibrated_volterra=np.loadtxt(NN/"pbs_validation_Volterra_calibrated_test_berall.txt"))
    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    for i,ax in enumerate(axes):
        for name,data in series.items():ax.semilogy(CASES,np.maximum(data[:,i],1e-10),".-",label=name)
        ax.set_title(("CH1","CH3","PBS")[i]);ax.set_xlabel("Experiment");ax.set_ylabel("Test BER");ax.grid();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(NN/"multiscale_quadratic_comparison.png",dpi=180);plt.close(fig)

def plot_saved_results():
    """只读取已有BER矩阵画图，不训练、不重新计算BER或选择参数。"""
    matrix={name:np.loadtxt(NN/f"multiscale_quadratic_{name}_test_berall.txt") for name in ("fused","short_only","long_only")}
    plot(matrix)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--inference",action="store_true",help="只回放冻结模型")
    mode.add_argument("--train",action="store_true",help="按原候选范围训练并校准")
    mode.add_argument("--plot-only",action="store_true",help="仅用已有矩阵生成BER图")
    args=parser.parse_args()
    if not THREADPOOL_AVAILABLE:
        print("未安装threadpoolctl：继续使用已设置的单线程BLAS环境，不影响均衡公式。",flush=True)
    with threadpool_limits(limits=8):
        if args.plot_only:plot_saved_results()
        elif args.inference or (LOCK.exists() and not args.train):
            evaluate(json.loads(LOCK.read_text(encoding="utf-8")))
        else:run()
