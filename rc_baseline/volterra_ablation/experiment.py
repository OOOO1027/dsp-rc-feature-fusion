from dsp_rc.portable import load_audit_index
"""Frozen finite Volterra ablation. Numeric loading stops at symbol 20000.
Run from rc_baseline with PYTHONPATH=src. No test mode exists.
"""
import argparse,csv,hashlib,itertools,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from dsp_rc.core import (VolterraLMS,align_parameters,align_segment,rms,windows,
                         cap_demodulate,array_hash)
from dsp_rc.experiment import file_hash,write_json,mask_for,ChannelPipeline
from dsp_rc.fusion import FusionModel,candidates
from dsp_rc.phase2 import enriched_scores
from dsp_rc.phase4 import per_symbol,paired_blocks
ROOT=Path(__file__).resolve().parents[2]
CH=['rx1','rx2']; OUTPUTS=CH+['xy']

def read(p): return json.loads(Path(p).read_text())
def config(): return read(ROOT/'rc_baseline/configs/index509.json')
def now(): return datetime.now(timezone.utc).isoformat()
def backend_configs():
    result=[]
    for name in ['linear80_reg','ridge_low','linear80_low','dual40_reg']:
        for seed in ([100,101,102] if name=='dual40_reg' else [100]):
            c=next(v.copy() for v in candidates() if v['name']==name)
            c['nonlinear_seed']=seed;result.append(c)
    return result

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    prior={}
    for base in [ROOT/'rc_baseline',ROOT/'audit_outputs/2026-10-16']:
        for p in base.rglob('*'):
            if p.is_file() and not any(x in p.parts for x in ['.venv','__pycache__','volterra_ablation']) and out not in p.parents:
                prior[str(p.relative_to(ROOT))]=file_hash(p)
    for base in [ROOT/'Data',ROOT/'NN数据20250320/NN数据20250320']:
        for p in base.glob('*.m'):prior[str(p.relative_to(ROOT))]=file_hash(p)
    for p in ROOT.glob('*.md'):prior[str(p.relative_to(ROOT))]=file_hash(p)
    write_json(out/'preserved_before.json',prior)
    write_json(out/'protocol.json',dict(utc=now(),index=509,used_symbol_ranges_0based=[[0,10000],[10000,20000]],guard=128,
        source_config=config(),frontends=['on','off'],backend_configs=backend_configs(),
        primary='2x2: on/off quadratic waveform terms x linear80_reg/dual40_reg; both 311 readout dimensions, ridge .01',
        anchors=['ridge_low','linear80_low'],nonlinear_seeds=[100,101,102],linear_seed=200,
        off='Refit W from zeros on same training prefix with quadratic term exactly absent; no reuse of co-trained W.',
        shared='alignment/raw/Tx RMS training-only shared across on/off; branch-specific wave/CAP/backend scales fitted on same training mask; fixed CAP phase0; same matrices across frontend conditions',
        alignment_gate='Training CAP correlation peak over lags -8..8 must be zero for every front/channel; otherwise stop before validation prediction. No tuning delay/phase.',
        search='No hyperparameter selection; exactly 12 frontend/backend/seed combinations, 24 channel models. Anchors fitted once per frontend/channel, not counted as independent seed replicates.',
        smoke='Synthetic unit checks then 3000 train/1000 validation symbols contained within original training; refit front on smoke training only; no selection.',
        bootstrap=dict(blocks=[256,512],replicates=2000,seed=404,method='paired moving blocks of predecoded validation losses; same positions across channels/seeds/four cells; average seed losses then bootstrap'),
        metrics=['complex MSE','fixed 2bit BER lower/upper','invalid count/rate','auto BER with I/Q widths'],
        effect='interaction=(dual_off-linear_off)-(dual_on-linear_on); negative favors increased incremental dual benefit off',
        promise_gate='Off dual must beat BOTH off anchors in mean MSE and not worsen mean BER upper in ALL three seeds; seed-mean paired 95% upper CI below0 for MSE and BER upper at BOTH block sizes against BOTH anchors; interaction MSE upper CI below0 at BOTH sizes. Else stop. No post-hoc expansion regardless of outcome.',
        inference_limit='Exploratory reused validation and fixed linear seed; not independent new-payload testing; no unique physical BER until illegal PAM4 contract fixed.',
        test_policy='Never parse numeric samples beyond80000 or symbols beyond20000; whole-file SHA256 only is allowed for integrity. No old test prediction artifacts opened.',
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in list((ROOT/'rc_baseline/src/dsp_rc').glob('*.py'))+list(Path(__file__).parent.glob('*.py'))}))
    write_json(out/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    print('Protocol frozen before data fitting:',file_hash(out/'protocol.json'),flush=True)

def assert_frozen(out):
    p=read(out/'protocol.json')
    for f,h in p['source_hashes'].items():assert file_hash(ROOT/f)==h,f
    return p

def load_prefix():
    nr=ROOT/'NN数据20250320/NN数据20250320'
    paths={'rx1':nr/'rxdata/rx1_PBS_NLTCP509.txt','rx2':nr/'rxdata/rx2_PBS_NLTCP509.txt',
           'tx':nr/'txdata/RF0.105_data36QAM.txt','i':nr/'v1.txt','q':nr/'v2.txt',
           'pam_i':nr/'decimal_PAM4_1.txt','pam_q':nr/'decimal_PAM4_2.txt'}
    audit=load_audit_index(ROOT)
    data={};manifest={}
    for k,p in paths.items():
        h=file_hash(p);assert h==audit[str(p)]['sha256']
        count=80000 if k in ['rx1','rx2','tx'] else 20000
        with p.open() as f:
            tokens=(v for line in f for v in line.split())
            data[k]=np.fromiter((float(v) for v in itertools.islice(tokens,count)),float,count=count)
        assert len(data[k])==count and np.isfinite(data[k]).all()
        manifest[k]=dict(path=str(p),sha256=h,numeric_prefix_count=count)
    y=data['i']+1j*data['q'];labels=np.column_stack([data['pam_i'],data['pam_q']]).astype(int)
    assert np.isin(labels,np.arange(4)).all()
    return data,y,labels,manifest

class LinearWaveLMS:
    """Same waveform support and one-pass update as VolterraLMS, no V term."""
    def __init__(self,taps,step):self.taps=taps;self.step=step
    def fit(self,x,d):
        self.W=np.zeros(self.taps);self.errors=[];h=self.taps//2
        for t in range(h,len(x)-h):
            u=x[t-h:t+h+1][::-1];e=d[t]-self.W@u
            self.W+=self.step*e*u;self.errors.append(e)
        self.errors=np.array(self.errors)
        if not np.isfinite(self.W).all():raise FloatingPointError('linear waveform diverged')
        return self
    def predict(self,x):return windows(x,self.taps)[:,::-1]@self.W

class Frontend:
    def __init__(self,channel,mode,c):self.channel=channel;self.mode=mode;self.c=c
    def fit(self,raw,tx,alignment):
        start=time.perf_counter();g=self.c['guard_symbols']*4
        self.alignment=alignment.copy();a=align_segment(raw,alignment['lag'],alignment['polarity'])
        self.rx_scale=rms(a[g:-g]);self.tx_scale=rms(tx[g:-g]);v=self.c['frontend'][self.channel]
        self.filter=VolterraLMS(*v) if self.mode=='on' else LinearWaveLMS(v[0],v[1])
        nt=min(self.c['frontend_train_samples'],len(raw))
        self.filter.fit(a[:nt]/self.rx_scale,tx[:nt]/self.tx_scale)
        wave=self.filter.predict(a/self.rx_scale);self.wave_scale=rms(wave[g:-g])
        z=cap_demodulate(wave/self.wave_scale,0,4,**self.c['cap']);m=mask_for(len(z),128)
        self.cap_scale=rms(z[m]);self.fit_seconds=time.perf_counter()-start
        return self
    def preprocess(self,raw,start_symbol):
        a=align_segment(raw,self.alignment['lag'],self.alignment['polarity'])/self.rx_scale
        wave=self.filter.predict(a)/self.wave_scale
        z=cap_demodulate(wave,start_symbol*4,4,**self.c['cap'])/self.cap_scale
        assert len(z)*4==len(raw) and np.isfinite(z).all()
        return wave,z
    def digest(self):
        return array_hash(self.filter.W,getattr(self.filter,'V',np.zeros(0)),np.array([self.rx_scale,self.tx_scale,self.wave_scale,self.cap_scale,self.alignment['lag'],self.alignment['polarity']]))
    def save(self,path):
        np.savez_compressed(path,channel=self.channel,mode=self.mode,config=json.dumps(self.c),alignment=json.dumps(self.alignment),W=self.filter.W,V=getattr(self.filter,'V',np.zeros(0)),errors=self.filter.errors,scales=[self.rx_scale,self.tx_scale,self.wave_scale,self.cap_scale])
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            f=cls(str(a['channel']),str(a['mode']),json.loads(str(a['config'])));f.alignment=json.loads(str(a['alignment']))
            v=f.c['frontend'][f.channel];f.filter=VolterraLMS(*v) if f.mode=='on' else LinearWaveLMS(v[0],v[1]);f.filter.W=a['W'];f.filter.errors=a['errors']
            if f.mode=='on':f.filter.V=a['V']
            f.rx_scale,f.tx_scale,f.wave_scale,f.cap_scale=map(float,a['scales'])
            return f

def smoke(out):
    assert_frozen(out);assert not (out/'smoke.json').exists()
    a,y,labels,_=load_prefix();c=config();m=mask_for(1000,128);results={}
    for ch in CH:
        alignment=align_parameters(a[ch][:10000],a['tx'][:10000],64)
        for mode in ['on','off']:
            f=Frontend(ch,mode,c).fit(a[ch][:12000],a['tx'][:12000],alignment)
            ztr=f.preprocess(a[ch][:12000],0)[1];zva=f.preprocess(a[ch][12000:16000],3000)[1]
            cfg=next(v for v in backend_configs() if v['name']=='dual40_reg')
            model=FusionModel(cfg).fit(ztr,y[:3000],mask_for(3000,128));p=model.predict(zva)
            key=mode+'_'+ch;model.save(out/f'smoke_{key}.npz');f.save(out/f'smoke_front_{key}.npz')
            np.testing.assert_array_equal(p,FusionModel.load(out/f'smoke_{key}.npz').predict(zva))
            np.testing.assert_array_equal(zva,Frontend.load(out/f'smoke_front_{key}.npz').preprocess(a[ch][12000:16000],3000)[1])
            results[key]=dict(finite=bool(np.isfinite(p).all()),symbols=len(p),scored=int(m.sum()))
    write_json(out/'smoke.json',dict(results=results,scope='3000/1000 entirely in original train; interface only; no candidate selection'))
    print('Four smoke paths passed',flush=True)

def run(out):
    protocol=assert_frozen(out);assert (out/'smoke.json').exists();assert not (out/'run_started.json').exists()
    write_json(out/'run_started.json',dict(utc=now(),protocol_sha256=file_hash(out/'protocol.json')))
    a,y,labels,manifest=load_prefix();write_json(out/'input_manifest.json',manifest)
    c=config();mask=mask_for(10000,128);models=out/'models';models.mkdir();fronts={};z={};frontstats={};parity={}
    # Fit and verify ALL frontends on train before any validation inference.
    for ch in CH:
        alignment=align_parameters(a[ch][:10000],a['tx'][:10000],64)
        old=ChannelPipeline.load(ROOT/f'rc_baseline/runs/index509_fixed_v1/model_{ch}.npz',ch,c)
        for mode in ['on','off']:
            key=mode+'_'+ch;f=Frontend(ch,mode,c).fit(a[ch][:40000],a['tx'][:40000],alignment);fronts[key]=f
            wave,tr=f.preprocess(a[ch][:40000],0);z[key]={'train':tr};t=np.flatnonzero(mask)
            corrs={str(lag):float(abs(np.vdot(tr[t+lag],y[t]))/np.sqrt(np.sum(abs(tr[t+lag])**2)*np.sum(abs(y[t])**2))) for lag in range(-8,9)}
            peak=int(max(corrs,key=corrs.get))
            frontstats[key]=dict(alignment=alignment,scales=[f.rx_scale,f.tx_scale,f.wave_scale,f.cap_scale],fit_seconds=f.fit_seconds,train_correlations=corrs,peak_lag=peak,
                linear_parameters=int(f.filter.W.size),quadratic_parameters=(int(v:=c['frontend'][ch][2])*(v+1)//2 if mode=='on' else 0),updates=len(f.filter.errors),wave_train_mse=float(np.mean((wave[512:-512]-a['tx'][512:39488]/f.tx_scale/f.wave_scale)**2)))
            if peak!=0:
                write_json(out/'alignment_blocked.json',frontstats);raise RuntimeError('Training alignment gate failed; no validation evaluation allowed')
            f.save(models/f'frontend_{key}.npz')
            if mode=='on':
                np.testing.assert_array_equal(f.filter.W,old.frontend.W);np.testing.assert_array_equal(f.filter.V,old.frontend.V)
                np.testing.assert_array_equal(tr,old.preprocess(a[ch][:40000],0)[1]);parity[ch]='W,V,training CAP exactly equal to saved baseline'
        assert fronts['on_'+ch].rx_scale==fronts['off_'+ch].rx_scale
        assert not np.array_equal(fronts['on_'+ch].filter.W,fronts['off_'+ch].filter.W)
    write_json(out/'train_frontend_checks.json',dict(stats=frontstats,original_frontend_parity=parity,all_training_lag0=True))
    for key,f in fronts.items():
        digest=f.digest();start=time.perf_counter();wave,va=f.preprocess(a[f.channel][40000:80000],10000);secs=time.perf_counter()-start;z[key]['validation']=va
        np.testing.assert_array_equal(va,Frontend.load(models/f'frontend_{key}.npz').preprocess(a[f.channel][40000:80000],10000)[1]);assert digest==f.digest()
        frontstats[key]['validation_preprocess_seconds']=secs
        for split in ['train','validation']:
            q=z[key][split][mask];frontstats[key][split+'_symbol_stats']=dict(mean_real=float(q.real.mean()),mean_imag=float(q.imag.mean()),power=float(np.mean(abs(q)**2)),amplitude_quantiles=np.quantile(abs(q),[.5,.9,.99,1]).tolist())
    write_json(out/'frontend_results.json',frontstats)
    predictions={};rows=[];matrix_hashes={}
    for mode in protocol['frontends']:
        for cfg in protocol['backend_configs']:
            name=cfg['name'];seed=cfg['nonlinear_seed'];key=f'{mode}_{name}_s{seed}';p={s:{} for s in ['train','validation']};stats={};timing={}
            for ch in CH:
                m=FusionModel(cfg).fit(z[mode+'_'+ch]['train'],y[:10000],mask);digest=m.digest();m.save(models/f'{key}_{ch}.npz');stats[ch]=m.stats
                hashes={k:array_hash(r.A,r.Win) for k,r in [('linear',m.linear),('nonlinear',m.nonlinear)] if r is not None}
                hk=f'{name}_s{seed}_{ch}'
                if mode=='on':matrix_hashes[hk]=hashes
                else:assert matrix_hashes[hk]==hashes
                for split in p:
                    start=time.perf_counter();p[split][ch]=m.predict(z[mode+'_'+ch][split]);timing[ch+'_'+split]=time.perf_counter()-start
                np.testing.assert_array_equal(p['validation'][ch],FusionModel.load(models/f'{key}_{ch}.npz').predict(z[mode+'_'+ch]['validation']));assert digest==m.digest()
            row=dict(key=key,frontend=mode,config=cfg,stats=stats,inference_seconds=timing)
            for split,start in [('train',0),('validation',10000)]:
                p[split]['xy']=(p[split]['rx1']+p[split]['rx2'])/2
                row[split]={ch:enriched_scores(p[split][ch],y[start:start+10000],labels[start:start+10000],mask) for ch in OUTPUTS}
                for ch in OUTPUTS:predictions[f'{key}_{split}_{ch}']=p[split][ch]
            rows.append(row);print(key,'validation MSE',np.mean([row['validation'][ch]['symbol_mse'] for ch in OUTPUTS]),flush=True)
    np.savez_compressed(out/'predictions.npz',**predictions,mask=mask,train_symbol_ids=np.arange(10000),validation_symbol_ids=np.arange(10000,20000))
    write_json(out/'results.json',rows);write_json(out/'matrix_hashes.json',matrix_hashes)
    losses={}
    for row in rows:
        key=row['key'];losses[key]=np.mean([per_symbol(predictions[f'{key}_validation_{ch}'],y[10000:],labels[10000:],mask) for ch in OUTPUTS],axis=0)
    contrasts={};seed_effects={}
    def contrast(name,delta):
        contrasts[name]={str(b):paired_blocks(delta,b,2000,404) for b in [256,512]}
    dualmean={mode:np.mean([losses[f'{mode}_dual40_reg_s{s}'] for s in [100,101,102]],axis=0) for mode in ['on','off']}
    for mode in ['on','off']:
        for base in ['linear80_reg','ridge_low','linear80_low']:
            contrast(mode+'_dual_minus_'+base,dualmean[mode]-losses[f'{mode}_{base}_s100'])
            seed_effects[mode+'_dual_minus_'+base]=[(losses[f'{mode}_dual40_reg_s{s}']-losses[f'{mode}_{base}_s100']).mean(axis=0).tolist() for s in [100,101,102]]
    interaction=(dualmean['off']-losses['off_linear80_reg_s100'])-(dualmean['on']-losses['on_linear80_reg_s100']);contrast('interaction',interaction)
    seed_effects['interaction']=[((losses[f'off_dual40_reg_s{s}']-losses['off_linear80_reg_s100'])-(losses[f'on_dual40_reg_s{s}']-losses['on_linear80_reg_s100'])).mean(axis=0).tolist() for s in [100,101,102]]
    for base in ['linear80_reg','ridge_low','linear80_low']:contrast(base+'_off_minus_on',losses[f'off_{base}_s100']-losses[f'on_{base}_s100'])
    contrast('dual_off_minus_on',dualmean['off']-dualmean['on'])
    contrast('off_dual_minus_on_linear80_low',dualmean['off']-losses['on_linear80_low_s100'])
    gate=True
    for base in ['ridge_low','linear80_low']:
        key='off_dual_minus_'+base
        gate &= all(x[0]<0 and x[1]<=0 for x in seed_effects[key])
        gate &= all(np.all(np.array(v['CI95'])[:,1]<0) for v in contrasts[key].values())
    gate &= all(v['CI95'][0][1]<0 for v in contrasts['interaction'].values())
    write_json(out/'paired_block_results.json',contrasts);write_json(out/'seed_effects.json',seed_effects)
    write_json(out/'decision.json',dict(utc=now(),promising_for_new_payload=bool(gate),search_expanded=False,test_evaluated=False,interaction_mean=interaction.mean(axis=0).tolist(),stop_current_experiment=True,interpretation='Finite experiment complete; no automatic follow-on training. Gate is exploratory on reused validation.'))
    old=read(out/'preserved_before.json');assert all(file_hash(ROOT/k)==h for k,h in old.items());assert_frozen(out)
    write_json(out/'verification.json',dict(old_files_unchanged=len(old),protocol_sha256=file_hash(out/'protocol.json'),numeric_samples_max=80000,numeric_symbols_max=20000,test_predictions=False,all_front_training_lags0=True,on_front_exact_historical_train_parity=True,all_models_reload_exact=True,on_off_matrices_equal=True,off_quadratic_terms=0,backend_channel_models=24,frontend_models=4))
    print('Finite run complete. Promise gate:',bool(gate),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','run']);p.add_argument('--out',type=Path,default=Path('runs/phase5_volterra_ablation'));a=p.parse_args();globals()[a.mode](a.out.resolve())
if __name__=='__main__':main()
