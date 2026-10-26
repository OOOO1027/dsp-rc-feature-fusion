"""Matched local-window ridge residual baselines for short and dual RC outputs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import rx_neural_dual_rc_lms as base

ROOT=Path(r'C:\Users\lenovo\Desktop\DSP exp\完整实验执行_20261025')
OUT=ROOT/'06_linear_residual';OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=(1e-5,1e-4,1e-3,.01,.1,1.)
RADIUS=14
VARIANTS=('short_linear_residual','dual_linear_residual')

def design(short,long,variant,start,stop,scale):
    def one(wave):
        padded=np.pad(np.asarray(wave,dtype=np.float64)/scale,(RADIUS,RADIUS),mode='edge')
        return np.lib.stride_tricks.sliding_window_view(padded,2*RADIUS+1)[start:stop]
    a=one(short)
    if variant=='short_linear_residual':return np.asarray(a)
    return np.column_stack((a,one(long)))

def training(channel,variant,lock,scale):
    path=OUT/f'CH{1 if channel==0 else 3}_{variant}.npz'
    if path.exists():
        with np.load(path) as d:
            return dict(coefficient=d['coefficient'],bias=float(d['bias']),mean=d['mean'],
                        std=d['std'],alpha=float(d['alpha']),val_mse=float(d['val_mse']),
                        train_seconds=float(d['train_seconds']),feature_count=int(d['feature_count']))
    began=time.perf_counter();xs=[];ys=[];valid=[]
    teacher=base.public.COMMON.WAVEFORM
    for case in base.existing.CASES:
        pair=base.prefix_pair(case,channel,lock)
        short,long=pair[:,0],pair[:,1]
        xs.append(design(short,long,variant,*base.TRAIN,scale))
        ys.append((teacher[base.TRAIN[0]:base.TRAIN[1]]-short[base.TRAIN[0]:base.TRAIN[1]])/scale/1.02)
        valid.append((design(short,long,variant,*base.VALID,scale),
                      short[base.VALID[0]:base.VALID[1]],
                      teacher[base.VALID[0]:base.VALID[1]]))
    x=np.concatenate(xs);y=np.concatenate(ys)
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),1e-8)
    z=(x-mean)/std
    ym=y.mean();gram=z.T@z/len(z);right=z.T@(y-ym)/len(z)
    candidates=[]
    for alpha in ALPHAS:
        normalized=np.linalg.solve(gram+alpha*np.eye(z.shape[1]),right)
        coefficient=normalized/std
        bias=float(ym-mean@coefficient)
        error=0.;count=0
        for vx,short,target in valid:
            pred=short+(vx@coefficient+bias)*scale
            error+=float(np.sum((pred-target)**2));count+=len(pred)
        candidates.append((error/count,alpha,coefficient,bias))
    val_mse,alpha,coefficient,bias=min(candidates,key=lambda item:item[0])
    elapsed=time.perf_counter()-began
    np.savez_compressed(path,coefficient=coefficient,bias=bias,mean=mean,std=std,
                        alpha=alpha,val_mse=val_mse,train_seconds=elapsed,feature_count=len(coefficient))
    return dict(coefficient=coefficient,bias=bias,mean=mean,std=std,alpha=alpha,
                val_mse=val_mse,train_seconds=elapsed,feature_count=len(coefficient))

def predict(short,long,variant,model,scale,chunk=16384):
    result=np.empty(len(short))
    for start in range(0,len(short),chunk):
        stop=min(start+chunk,len(short))
        x=design(short,long,variant,start,stop,scale)
        result[start:stop]=short[start:stop]+(x@model['coefficient']+model['bias'])*scale
    return result

def main():
    base.public.COMMON.init_worker()
    lock=base.selected();scale=base.scale()
    models={(channel,variant):training(channel,variant,lock,scale)
            for channel in range(2) for variant in VARIANTS}
    arr={variant:np.full((19,3),np.nan) for variant in VARIANTS}
    times={variant:0. for variant in VARIANTS}
    for case in base.existing.CASES:
        cache=OUT/f'case{case:02d}.json'
        if cache.exists():
            saved=json.loads(cache.read_text(encoding='utf-8'))
            for variant in VARIANTS:
                arr[variant][case-1]=saved['test'][variant]
                times[variant]+=sum(saved['wave_inference_seconds'][variant])
            continue
        symbols={variant:[] for variant in VARIANTS};elapsed={variant:[] for variant in VARIANTS}
        for channel in range(2):
            short,long=base.full_pair(case,channel,lock)
            for variant in VARIANTS:
                begin=time.perf_counter()
                wave=predict(short,long,variant,models[(channel,variant)],scale)
                elapsed[variant].append(time.perf_counter()-begin)
                symbols[variant].append(base.public.backend(wave,channel))
        saved={'case':case,'test':{},'wave_inference_seconds':elapsed}
        for variant,pair in symbols.items():
            ch=[float(base.public.rates(base.public.channel_counts(s,base.public.COMMON.REFERENCES,base.TEST),base.TEST)[0]) for s in pair]
            pbs=float(base.public.rates(base.public.weighted_counts(*pair,np.array([.59]),base.TEST,
                                                              base.public.COMMON.REFERENCES),base.TEST)[0,0])
            arr[variant][case-1]=ch+[pbs]
            saved['test'][variant]=arr[variant][case-1].tolist()
            times[variant]+=sum(elapsed[variant])
        cache.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
        print('linear residual case',case,flush=True)
    rows=[]
    for variant in VARIANTS:
        x=arr[variant];np.savetxt(OUT/f'{variant}_test_berall.txt',x,fmt='%.16e')
        rows.append([variant,*x.mean(axis=0),models[(0,variant)]['feature_count'],
                     models[(0,variant)]['alpha'],models[(1,variant)]['alpha'],
                     np.mean([models[(ch,variant)]['val_mse'] for ch in range(2)]),
                     sum(models[(ch,variant)]['train_seconds'] for ch in range(2)),times[variant]])
    with (ROOT/'results'/'tables'/'linear_residual_baselines.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','CH1_BER','CH3_BER','PBS_BER','feature_count_per_channel',
                                    'CH1_alpha','CH3_alpha','mean_validation_wave_MSE','train_seconds',
                                    'wave_inference_seconds_all_38']);w.writerows(rows)
    print(json.dumps({r[0]:r[3] for r in rows},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
