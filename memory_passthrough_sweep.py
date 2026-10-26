"""双通道记忆参数和CH3线性直通抽头的受控扫描，不改公共DSP。"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ[key]="1"
import json,hashlib,time,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import ch3_capacity_sweep as prior
from quadratic_rc_equalizer import build_rc_features
from ch3_polynomial_readout import fit_models,predict_models,save_candidates,load_candidates,load_old_model
from rx_soft_gate_rc_lms import robust_validation_score

ROOT=prior.ROOT
NN=prior.NN
CACHE=NN/"memory_passthrough_models_v2"
FINAL=NN/"memory_passthrough_selected_models"
LOCK=NN/"memory_passthrough_selection.json"
SEARCH=prior.SEARCH_CASES
CONFIRM=prior.CONFIRM_CASES
ALPHAS=(.0001,.001,.0031622776601683794,.01,.1)
ALPHAS1=(1e-6,1e-5)+ALPHAS
RADII3=(.5,.9,1.1,1.3)
GAMMAS3=(.2,.5,.8,1.)
RADII1=(.7,.9,1.1)
GAMMAS1=(.5,.8,1.)
DIRECT=(33,49,65,97)

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def features(wave,target,params,normalization=None):
    """额外抽头仅进入读出层；储备池继续用原taps，随机矩阵和状态不变。"""
    base,scale,fit,norm=build_rc_features(wave,target,params,prior.TRAIN_END,normalization)
    taps=params.get("passthrough_taps",params["taps"])
    if taps<params["taps"] or taps%2!=1: raise ValueError("直通抽头须为不小于原窗口的奇数")
    if taps!=params["taps"]:
        half=taps//2
        window=np.lib.stride_tricks.sliding_window_view(np.pad(np.asarray(wave).real/norm["input_rms"],(half,half)),taps)
        # 沿用原训练区输入尺度，不使用验证或测试统计量。
        window=window/norm["input_window_scale"]
        base=np.column_stack((base[:,:params["nodes"]],window,np.zeros_like(window),np.ones(len(wave))))
    # 抽头阶段所有候选统一拟合边界，避免长窗口跨入验证区或比较样本数不同。
    margin=params.get("fit_margin",params["taps"]//2)
    return base,scale,slice(margin+1,prior.TRAIN_END-margin),norm

def old_model(case,ch):
    if ch==1:
        models,params,norm,_,_=load_candidates(prior.SELECTED/f"case{case:02d}_CH3_selected.npz")
        return models[0],params,norm
    return load_old_model(NN/f"quadratic_rc_models/case{case:02d}_CH1_quadratic_ridge.npz")

def score(wave,ch):
    symbols=prior.original.backend(wave,ch)
    value,blocks=robust_validation_score(symbols,tuple(x[:prior.TRAIN_SYMBOLS] for x in prior.original.common.REFERENCES),prior.SEGMENTS)
    return dict(score=float(value),segment_bers=blocks)

def prefix(case,ch):
    return prior.original.common.load(f"rxdata/rx{ch+1}_{case}.txt")[:prior.TRAIN_WAVE]

def key(params):
    return hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:16]

def job(case,ch,params,alphas=ALPHAS):
    path=CACHE/f"CH{1 if ch==0 else 3}_case{case:02d}_{key(params)}.npz"
    if path.exists():
        models,_,_,history,_=load_candidates(path)
        if all(any(m.alpha==a for m in models) for a in alphas):return path,history
    wave=prefix(case,ch)
    base,scale,fit,norm=features(wave,prior.original.common.WAVEFORM[:prior.TRAIN_WAVE],params)
    models,info=fit_models(base[fit],prior.original.common.WAVEFORM[fit]/scale,nodes=40,degree=2,target_scale=scale,alphas=alphas,verbose=False)
    output=predict_models(base,models)
    history=[dict(score(output[:,i],ch),alpha=m.alpha) for i,m in enumerate(models)]
    save_candidates(path,models,params,norm,history,info)
    return path,history

def baseline_score(case,ch):
    path=CACHE/f"baseline_CH{ch}_case{case:02d}.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    model,params,norm=old_model(case,ch)
    base,_,_,_=features(prefix(case,ch),None,params,norm)
    result=score(predict_models(base,[model])[:,0],ch)
    save(path,result)
    return result

def search(ch,configs,label):
    rows=[]
    alphas=ALPHAS1 if ch==0 else ALPHAS
    for index,params in enumerate(configs):
        histories=[job(case,ch,params,alphas)[1] for case in SEARCH]
        for alpha in alphas:
            entries=[next(x for x in h if x["alpha"]==alpha) for h in histories]
            rows.append(dict(params=params,alpha=alpha,score=float(np.mean([x["score"] for x in entries]))))
        best=min(rows[-len(alphas):],key=lambda x:x["score"])
        print(f"{label} {index+1}/{len(configs)} radius={params['spectral_radius']} gamma={params['gamma']} direct={params.get('passthrough_taps',params['taps'])} alpha={best['alpha']:.8g} score={best['score']:.8g}",flush=True)
    best=min(rows,key=lambda x:x["score"])
    save(CACHE/f"{label}_grid.json",dict(best=best,grid=rows))
    return best

def entries(candidate,ch,cases=CONFIRM):
    result=[]
    for case in cases:
        _,history=job(case,ch,candidate["params"],(candidate["alpha"],))
        result.append(next(x for x in history if x["alpha"]==candidate["alpha"]))
    return result

def confirm(candidate,ch,reference):
    result=entries(candidate,ch)
    return dict(entries=result,acceptance=prior.accepts(result,reference))

def candidate_model(case,ch,candidate):
    path,_=job(case,ch,candidate["params"],(candidate["alpha"],))
    models,params,norm,_,_=load_candidates(path)
    return next(m for m in models if m.alpha==candidate["alpha"]),params,norm

def configuration(base,radius,gamma,**extra):
    return dict(base,spectral_radius=radius,gamma=gamma,nodes=40,seed=100,**extra)

def protection():
    names=("quadratic_rc_equalizer.py","fusion_prefix_backend.py","receiver_metrics.py","rclzl_w2s.py","rx_quadratic_rc_lms.py","ch3_capacity_sweep.py","ch3_polynomial_readout.py")
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}

def run():
    prior.initialize();CACHE.mkdir(exist_ok=True);FINAL.mkdir(exist_ok=True)
    protected=protection()
    manifest=dict(alphas=ALPHAS,alphas_CH1=ALPHAS1,radii3=RADII3,gammas3=GAMMAS3,radii1=RADII1,gammas1=GAMMAS1,direct=DIRECT,search=SEARCH,confirm=CONFIRM,protected=protected,
        module_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        previous_selection_sha=hashlib.sha256(prior.LOCK.read_bytes()).hexdigest(),
        training_end=prior.TRAIN_END,segments=prior.SEGMENTS)
    signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=CACHE/"manifest.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8"))["signature"]!=signature:raise ValueError("缓存配置不同，请另设缓存目录")
    save(mp,dict(signature=signature,configuration=manifest))
    start=time.perf_counter()
    _,base3,_=old_model(1,1);_,base1,_=old_model(1,0)
    refs3=[baseline_score(case,1) for case in CONFIRM]
    refs1=[baseline_score(case,0) for case in CONFIRM]
    # 第一步只改变记忆参数和岭强度，节点、储备池输入窗口不变。
    memory=search(1,[configuration(base3,r,g) for r in RADII3 for g in GAMMAS3],"CH3_memory")
    mc=confirm(memory,1,refs3)
    current=memory if mc["acceptance"]["accepted"] else None
    source=current["params"] if current else base3
    # 第二步只扩展线性直通窗口；最大窗口边界统一用于此阶段所有候选。
    direct=search(1,[dict(source,passthrough_taps=t,fit_margin=max(DIRECT)//2) for t in DIRECT],"CH3_passthrough")
    dc=confirm(direct,1,refs3)
    preceding=mc["entries"] if current else refs3
    dc["versus_previous"]=prior.accepts(dc["entries"],preceding)
    if dc["acceptance"]["accepted"] and dc["versus_previous"]["accepted"]:current=direct
    # CH1独立微调，不按PBS测试结果选通道参数。
    first=search(0,[configuration(base1,r,g) for r in RADII1 for g in GAMMAS1],"CH1_memory")
    fc=confirm(first,0,refs1)
    chosen1=first if fc["acceptance"]["accepted"] else None
    selection=dict(signature=signature,CH3_memory=memory,CH3_memory_confirmation=mc,CH3_passthrough=direct,CH3_passthrough_confirmation=dc,
        CH1_candidate=first,CH1_confirmation=fc,selected_CH1=chosen1,selected_CH3=current,
        protected=protected,test_used_for_selection=False,seconds_before_test=time.perf_counter()-start,
        caveat="改善只支持记忆参数或直通路径有帮助，不证明缺记忆是唯一瓶颈；增大gamma并不等于延长记忆")
    save(LOCK,selection)
    for case in range(1,20):
        for ch,candidate in enumerate((chosen1,current)):
            model,params,norm=candidate_model(case,ch,candidate) if candidate else old_model(case,ch)
            save_candidates(FINAL/f"case{case:02d}_CH{1 if ch==0 else 3}.npz",[model],params,norm,[],dict(frozen=True))
    print("参数已按验证区锁定，开始十九组冻结测试",flush=True)
    result=evaluate(selection)
    assert protection()==protected
    result["protected_files_unchanged"]=True;result["elapsed_seconds"]=time.perf_counter()-start
    save(NN/"memory_passthrough_summary.json",result)

def symbols(case,ch,model,params,norm):
    wave=prior.original.common.load(f"rxdata/rx{ch+1}_{case}.txt")
    if len(wave)!=204800:raise ValueError("波形长度须为204800")
    base,_,_,_=features(wave,None,params,norm)
    result=prior.original.backend(predict_models(base,[model])[:,0],ch)
    if len(result)!=51200:raise ValueError("CAP符号长度须为51200")
    return result

def evaluate(selection):
    if not LOCK.exists() or json.loads(LOCK.read_text(encoding="utf-8"))!=selection:raise ValueError("须加载已冻结选择")
    prior.initialize()
    names=("baseline","CH3_only","CH1_only","both")
    matrices={kind:{name:np.empty((19,3)) for name in names} for kind in ("legacy","test")}
    for case in range(1,20):
        before=[symbols(case,ch,*old_model(case,ch)) for ch in range(2)]
        after=[]
        for ch in range(2):
            models,params,norm,_,_=load_candidates(FINAL/f"case{case:02d}_CH{1 if ch==0 else 3}.npz")
            after.append(symbols(case,ch,models[0],params,norm))
        options=(before,(before[0],after[1]),(after[0],before[1]),after)
        for name,pair in zip(names,options):
            legacy,test,_=prior.original.common.metrics(pair,15488)
            matrices["legacy"][name][case-1]=legacy;matrices["test"][name][case-1]=test
        print(f"冻结测试 {case}/19 CH1/CH3/PBS={matrices['test']['both'][case-1].tolist()}",flush=True)
    for kind in matrices:
        expected=np.loadtxt(NN/("ch3_capacity_test_berall.txt" if kind=="test" else "ch3_capacity_berall.txt"))
        np.testing.assert_array_equal(matrices[kind]["baseline"],expected)
        for name,value in matrices[kind].items():
            assert value.shape==(19,3) and np.isfinite(value).all()
            np.savetxt(NN/f"memory_passthrough_{name}_{kind}_berall.txt",value,fmt="%.16e")
    summary=dict(selection=selection,means={name:{kind:matrices[kind][name].mean(axis=0).tolist() for kind in matrices} for name in names},baseline_exactly_reproduced=True)
    for name in names:print(f"{name} 平均测试BER={summary['means'][name]['test']}",flush=True)
    save(NN/"memory_passthrough_summary.json",summary)
    plot(matrices["test"])
    return summary

def plot(matrices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series=dict(matrices,volterra=np.loadtxt(NN/"quadratic_rc_volterra_lms_test_berall.txt"))
    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    for i,ax in enumerate(axes):
        for name,data in series.items():ax.semilogy(range(1,20),np.maximum(data[:,i],1e-10),".-",label=name)
        ax.set_title(("CH1","CH3","PBS")[i]);ax.set_xlabel("Experiment");ax.set_ylabel("Test BER");ax.grid();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(NN/"memory_passthrough_comparison.png",dpi=180);plt.close(fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--inference",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=8):
        if args.inference:evaluate(json.loads(LOCK.read_text(encoding="utf-8")))
        else:run()
