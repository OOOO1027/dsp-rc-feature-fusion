"""残差级联RC波形均衡，保留上一版并行融合及所有原始实验文件。"""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,hashlib
import numpy as np
import rx_dual_rc_fusion_lms as common
from fusion_prefix_backend import prefix_cap,prefix_lms
from residual_rc_equalizer import residual_rc_equalize

ROOT=Path(__file__).resolve().parent
NN_ROOT=common.NN_ROOT
SCHEMES=("linear_only","residual")

def init_worker():
    common.init_worker()

def run_one_case(task):
    case,train_ratio,native,verbose=task
    train_symbols=int(common.DATANO*train_ratio)
    residual_end=4*train_symbols
    linear_end=int(residual_end*common.BRANCH_FRACTION)
    test_start=max(common.BER_START,train_symbols+128)
    outputs={name:[] for name in SCHEMES}
    records=[]
    for channel in (1,2):
        received=common.load(f"rxdata/rx{channel}_{case}.txt")
        if len(received)!=common.DATANO*4:
            raise ValueError("接收波形必须为204800点")
        corrected,linear,correction,record=residual_rc_equalize(received,common.WAVEFORM,
            linear_train_samples=linear_end,residual_train_samples=residual_end,
            linear_params=common.WAVE_RC[channel-1],native=native,verbose=verbose)
        records.append(dict(record,channel=1 if channel==1 else 3))
        for name,wave in (("linear_only",linear),("residual",corrected)):
            symbols=prefix_cap(wave,4,common.DATANO,training_count=common.TRAIN_SYMBOL)
            recovered=prefix_lms(symbols,common.TARGET,common.SYMBOL_LMS[channel-1],
                                 common.TRAIN_SYMBOL,native)
            if len(recovered)!=common.DATANO:
                raise ValueError("符号输出必须与51200点理想参考对齐")
            outputs[name].append(recovered)
            if verbose:
                print(f"第{case}组 CH{1 if channel==1 else 3} {name}：CAP{symbols.shape}→LMS{recovered.shape}",flush=True)
    results={}
    for name in SCHEMES:
        legacy,test,weights=common.metrics(outputs[name],test_start)
        results[name]={"legacy":legacy.tolist(),"test":test.tolist(),"pbs_weights":weights}
    return case,results,records

def plot_results(matrices,train_ratio,show=False):
    import matplotlib
    if not show: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series={"Residual cascade":matrices["residual"],"Linear only":matrices["linear_only"]}
    summary_path=NN_ROOT/"dual_rc_fusion_summary.json"
    # 只加载统计口径和后端一致的前轮结果，避免混用历史数据。
    if summary_path.exists():
        previous=json.loads(summary_path.read_text(encoding="utf-8"))
        comparable=(previous["train_ratio"]==train_ratio
            and previous["symbol_lms"]==[list(x) for x in common.SYMBOL_LMS]
            and previous["wave_rc"]==list(common.WAVE_RC))
        if comparable:
            for name in ("projection","weighted_sum","fixed_weight","volterra_lms","pure_rc"):
                path=NN_ROOT/f"dual_rc_fusion_{name}_test_berall.txt"
                if path.exists(): series[name]=np.loadtxt(path)
    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    for column,ax in enumerate(axes):
        for name,m in series.items():
            ax.semilogy(range(1,20),np.maximum(m[:,column],np.finfo(float).tiny),label=name,marker=".")
        ax.set_title(("CH1","CH3","PBS")[column]);ax.set_xlabel("Experiment");ax.set_ylabel("Test BER")
        ax.grid(True);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(NN_ROOT/"residual_rc_comparison_test.png",dpi=180)
    if show: plt.show()
    plt.close(fig)

def run_all_experiments(train_ratio=.3,workers=4,native=False,verbose=True,show=False):
    if not 0<train_ratio<1: raise ValueError("训练比例必须位于0与1之间")
    paths=("rx_dual_rc_fusion_lms.py","fusion_layer.py","fusion_prefix_backend.py",
           "linear_rc_equalizer.py","rclzl_w2s.py","receiver_metrics.py","rx_parallel_rc_lms.py")
    protected={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    matrices={kind:{name:np.full((19,3),np.nan) for name in SCHEMES} for kind in ("legacy","test")}
    records=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for case,result,parameters in pool.map(run_one_case,
            [(n,train_ratio,native,verbose) for n in range(1,20)]):
            for kind in matrices:
                for name in SCHEMES:
                    matrices[kind][name][case-1]=result[name][kind]
                    np.savetxt(NN_ROOT/f"residual_rc_{name}_{kind}_berall.txt",
                               matrices[kind][name],fmt="%.16e")
            records.append({"case":case,"channels":parameters})
            print(f"第{case}组残差级联测试BER={result['residual']['test']}",flush=True)
    summary={"architecture":"linear_then_nonlinear_residual","train_ratio":train_ratio,
        "linear_train_samples":int(4*int(common.DATANO*train_ratio)*common.BRANCH_FRACTION),
        "residual_train_samples":4*int(common.DATANO*train_ratio),
        "test_slice":[max(common.BER_START,int(common.DATANO*train_ratio)+128),common.DATANO-common.BER_TAIL],
        "wave_rc":common.WAVE_RC,"symbol_lms":common.SYMBOL_LMS,
        "correction_target":"reference_minus_linear","correction_gain":1.,
        "reference_used_only_in_training":True,"means":{},"training":records,
        "protected_sha256":protected,
        "comparison_note":"残差级在30%训练前缀内拟合，并扣除级联前视保护；前轮并行非线性分支使用20%训练区，不能将结果变化仅归因于结构。"}
    for name in SCHEMES:
        summary["means"][name]={kind:matrices[kind][name].mean(axis=0).tolist() for kind in matrices}
        print(f"{name}平均测试CH1/CH3/PBS BER={summary['means'][name]['test']}",flush=True)
    for name,h in protected.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:
            raise RuntimeError("原文件发生意外变化："+name)
    np.savetxt(NN_ROOT/"residual_rc_test_berall.txt",matrices["test"]["residual"],fmt="%.16e")
    np.savetxt(NN_ROOT/"residual_rc_berall.txt",matrices["legacy"]["residual"],fmt="%.16e")
    (NN_ROOT/"residual_rc_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    plot_results(matrices["test"],train_ratio,show)
    return matrices,summary

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="线性主体与非线性残差RC级联")
    parser.add_argument("--train-ratio",type=float,default=.3)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--quiet",action="store_true")
    parser.add_argument("--show",action="store_true")
    args=parser.parse_args()
    run_all_experiments(args.train_ratio,args.workers,args.native,not args.quiet,args.show)
