"""Phase 2: predeclared validation-only diagnostics/search, frozen confirmation.

Keeps phase-1 code, configurations and result directories untouched.
"""
import argparse, copy, csv, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .core import (Reservoir,symbol_features,fit_ridge,array_hash,decode_nltcp,rms,align_segment,VolterraLMS,cap_demodulate)
from .experiment import (ChannelPipeline,load_assets,mask_for,scores,write_json,file_hash,check_config)


def candidate_list():
    base=dict(taps=25,nodes=80,radius=.9,leak=.7,ridge=1e-5,normalization='max',gain=1.,seed=100)
    changes=[('original',{}),('window57',{'taps':57}),('window115',{'taps':115}),
             ('nodes160',{'nodes':160}),('radius03',{'radius':.3}),('leak03',{'leak':.3}),
             ('ridge001',{'ridge':.01}),('rms_input',{'normalization':'rms'}),
             ('input025',{'gain':.25}),('window115_input025',{'taps':115,'gain':.25})]
    return [dict(base,**patch,name=name) for name,patch in changes]


def enriched_scores(z,y,labels,mask):
    m=scores(z,y,labels,mask);m['invalid_rate']=m['invalid_pam4_decisions']/(2*m['symbols'])
    return m


class RCModel:
    def __init__(self,config):self.config=config.copy()
    def fit(self,z,y,mask):
        c=self.config;t=time.perf_counter();x=symbol_features(z,c['taps'])
        self.scale=float(np.max(abs(x[mask]))) if c['normalization']=='max' else rms(x[mask])
        x=x/self.scale*c['gain'];self.reservoir=Reservoir(x.shape[1],c['nodes'],c['leak'],c['radius'],c['seed'])
        st,f,_=self.reservoir.transform(x);target=np.column_stack([y.real,y.imag])
        self.Wout=fit_ridge(f[mask],target[mask],c['ridge'])
        self.fit_seconds=time.perf_counter()-t
        prev=np.vstack([np.zeros((1,c['nodes'])),st[:-1]])
        activation=(st-(1-c['leak'])*prev)/c['leak']
        self.stats={'activation_abs_gt_099':float(np.mean(abs(activation[mask])>.99)),
                    'state_std_mean':float(st[mask].std(axis=0).mean()),
                    'feature_dim':f.shape[1],'trained_parameters':int(self.Wout.size),
                    'fixed_parameters':int(self.reservoir.A.size+self.reservoir.Win.size),
                    'input_dim':x.shape[1], 'readout_norm':float(np.linalg.norm(self.Wout)),
                    'gram_condition':float(np.linalg.cond(f[mask].T@f[mask]+c['ridge']*np.eye(f.shape[1])))}
        return self
    def predict(self,z):
        x=symbol_features(z,self.config['taps'])/self.scale*self.config['gain']
        states,features,_=self.reservoir.transform(x);o=features@self.Wout
        return o[:,0]+1j*o[:,1]
    def digest(self):return array_hash(self.reservoir.A,self.reservoir.Win,self.Wout,np.array([self.scale]))
    def save(self,path):
        np.savez_compressed(path,A=self.reservoir.A,Win=self.reservoir.Win,Wout=self.Wout,scale=self.scale)
    @classmethod
    def load(cls,path,c):
        m=cls(c);m.reservoir=Reservoir(2*c['taps'],c['nodes'],c['leak'],c['radius'],c['seed'])
        with np.load(path) as a:m.reservoir.A=a['A'];m.reservoir.Win=a['Win'];m.Wout=a['Wout'];m.scale=float(a['scale'])
        return m


def error_diagnostics(pred,target,mask):
    p=pred[mask];y=target[mask];g=np.vdot(p,y)/np.vdot(p,p)
    q,d=decode_nltcp(pred);truthq,_=decode_nltcp(target);idx=np.flatnonzero(mask)
    inv=d[idx]>3;current=q[idx]!=truthq[idx];previous=q[idx-1]!=truthq[idx-1]
    counts={}
    for a in [False,True]:
        for b in [False,True]:counts[f'previous_wrong_{a}_current_wrong_{b}']=int(np.sum(inv&(previous==a)&(current==b)))
    return {'output_rms':rms(p),'target_rms':rms(y),'mean_I':float(p.real.mean()),'mean_Q':float(p.imag.mean()),
            'optimal_diagnostic_gain_abs':float(abs(g)),'optimal_diagnostic_gain_phase_deg':float(np.angle(g,deg=True)),
            'mse_before_gain':float(np.mean(abs(p-y)**2)),'mse_after_diagnostic_gain':float(np.mean(abs(g*p-y)**2)),
            'error_abs_quantiles':np.quantile(abs(p-y),[.5,.95,.99,1]).tolist(),
            'nonfinite':int((~np.isfinite(p)).sum()),'illegal_transition_causes':counts}


def decoder_audit(y,labels,c):
    q,d=decode_nltcp(y);bad=[(np.flatnonzero(d[:,k]!=labels[:,k])+1).tolist() for k in range(2)]
    forbidden=[{'previous':p,'current':v,'decoded':abs(v-p//2)} for p in range(6) for v in range(6) if abs(v-p//2)>3]
    transitions=np.zeros((6,6),int)
    for k in range(2):np.add.at(transitions,(q[:-1,k],q[1:,k]),1)
    split={}
    for name,(a,b) in c['splits'].items():
        mask=mask_for(b-a,c['guard_symbols'])
        split[name]=enriched_scores(y[a:b],y[a:b],labels[a:b],mask)
    return {'mismatch_positions_1based_per_component':bad,'forbidden_transitions':forbidden,
            'ideal_transition_counts':transitions.tolist(),'ideal_split_scores':split,
            'bounds_formula':'Elegal/(4N) <= BER_2bit <= (Elegal+2*U)/(4N); U counts component decisions, not complex symbols',
            'invalid_rate_formula':'U/(2N)','boundary_first_decoder_default':[0,0],
            'first_labels':labels[0].tolist(),'last_decoded':d[-1].tolist(),'last_labels':labels[-1].tolist()}


def signal_stats(x):
    x=np.asarray(x);center=x-x.mean();p=np.mean(abs(center)**2)
    return {'rms':rms(x),'mean_real':float(x.real.mean()),'mean_imag':float(x.imag.mean()),
            'abs_p99':float(np.quantile(abs(x),.99)),'peak_abs':float(max(abs(x))),
            'kurtosis_magnitude':float(np.mean(abs(center)**4)/p**2),
            'acf_abs':[float(abs(np.vdot(center[:-k],center[k:]))/np.sum(abs(center)**2)) for k in [1,2,4,12,25,57]]}


def run_search(root,out):
    if (out/'search_complete.json').exists():raise RuntimeError('Search already completed; do not overwrite')
    out.mkdir(parents=True,exist_ok=True);base=root/'rc_baseline';old=base/'runs/index509_fixed_v1'
    c=json.loads((old/'config.json').read_text());candidates=candidate_list()
    plan={'utc':datetime.now(timezone.utc).isoformat(),'candidates':candidates,
          'selection':'per channel min validation fixed-2bit BER upper; ties by MSE, then readout dimension, then name',
          'seed_policy':'100 is selection/final seed; 101,102 are validation-only stability checks for original and winner, never selected',
          'diagnostics_not_candidates':['linear ridge taps25/57/115','training-only output gain','refit RX2 frontend without polarity correction'],
          'test_policy':'one later confirm command; evaluate original, selected and existing LMS only; no test-based changes',
          'xy_weights':[.5,.5],'source_hashes':{str(p.relative_to(base)):file_hash(p) for p in (base/'src/dsp_rc').glob('*.py')}}
    write_json(out/'protocol_before_search.json',plan)
    assets,manifest=load_assets(root,c);write_json(out/'input_manifest.json',manifest)
    y=assets['i']+1j*assets['q'];labels=np.column_stack([assets['pam_i'],assets['pam_q']]).astype(int)
    write_json(out/'decoder_audit.json',decoder_audit(y,labels,c))
    front={ch:ChannelPipeline.load(old/f'model_{ch}.npz',ch,c) for ch in ['rx1','rx2']}
    z={};prediction={};metrics={};diag={};models={};search=[];linear=[];robust=[];mask=mask_for(10000,c['guard_symbols'])
    for ch,p in front.items():
        z[ch]={};diag[ch]={};prediction[ch]={};metrics[ch]={};models[ch]={}
        for split,(a,b) in list(c['splits'].items())[:2]:
            raw=assets[ch][a*4:b*4];z[ch][split]=p.preprocess(raw,a)[1]
            r=p.transform(raw,a);prediction[ch][split]={k:r[k] for k in ['rc','lms','before']}
            yy=y[a:b];ll=labels[a:b]
            diag[ch][split]={'raw':signal_stats(raw),'cap':signal_stats(z[ch][split]),
                             'old_rc':error_diagnostics(r['rc'],yy,mask),'lms':error_diagnostics(r['lms'],yy,mask)}
            metrics[ch][split]={method:enriched_scores(v,yy,ll,mask) for method,v in prediction[ch][split].items()}
        # Explicit alignment/polarity provenance from frozen training calibration.
        diag[ch]['alignment']=p.alignment
        diag[ch]['scales']={k:getattr(p,k) for k in ['rx_scale','wave_scale','cap_scale','target_scale','input_scale']}
        # Controlled linear readout window test: same features/solver, no reservoir.
        for taps in [25,57,115]:
            x=symbol_features(z[ch]['train'],taps);scale=float(max(abs(x[mask]).ravel()))
            xt=np.column_stack([x/scale,np.ones(len(x))]);xv=np.column_stack([symbol_features(z[ch]['validation'],taps)/scale,np.ones(len(x))])
            w=fit_ridge(xt[mask],np.column_stack([y[:10000].real,y[:10000].imag])[mask],1e-5)
            o=xv@w;pred=o[:,0]+1j*o[:,1]
            linear.append({'channel':ch,'taps':taps,'metrics':enriched_scores(pred,y[10000:20000],labels[10000:20000],mask)})
        # A train-only scalar gain diagnoses gross output amplitude/phase error.
        oldtr=prediction[ch]['train']['rc'];g=np.vdot(oldtr[mask],y[:10000][mask])/np.vdot(oldtr[mask],oldtr[mask])
        diag[ch]['train_gain_applied_to_validation']={'abs':float(abs(g)),'phase_deg':float(np.angle(g,deg=True)),
            'metrics':enriched_scores(prediction[ch]['validation']['rc']*g,y[10000:20000],labels[10000:20000],mask)}
        for candidate in candidates:
            cc=copy.deepcopy(c);cc['rc']['window_symbols']=candidate['taps'];check_config(cc)
            model=RCModel(candidate).fit(z[ch]['train'],y[:10000],mask);h=model.digest()
            tr=model.predict(z[ch]['train']);t=time.perf_counter();va=model.predict(z[ch]['validation']);pt=time.perf_counter()-t
            assert model.digest()==h
            if candidate['name']=='original':np.testing.assert_allclose(va,prediction[ch]['validation']['rc'],atol=2e-10,rtol=1e-10)
            row={'channel':ch,'candidate':candidate,'fit_seconds':model.fit_seconds,'validation_inference_seconds':pt,'complexity':model.stats,
                 'train':enriched_scores(tr,y[:10000],labels[:10000],mask),'validation':enriched_scores(va,y[10000:20000],labels[10000:20000],mask)}
            search.append(row);models[ch][candidate['name']]=model
            print(ch,candidate['name'],'validation upper',row['validation']['ber_upper'],'auto',row['validation']['matlab_compatible_ber'],flush=True)
    # Polarity ablation: train-only refit of waveform front with no sign correction.
    p=front['rx2'];raw=assets['rx2'][:40000];tx=assets['tx'][:40000]
    f=VolterraLMS(*c['frontend']['rx2']).fit(raw[:10000]/p.rx_scale,tx[:10000]/p.tx_scale)
    corrected=p.frontend.predict(-raw/p.rx_scale);uncorrected=f.predict(raw/p.rx_scale)
    diag['rx2']['polarity_equivalence']={'linear_weight_sign_relative_error':float(np.linalg.norm(f.W+p.frontend.W)/np.linalg.norm(p.frontend.W)),
        'quadratic_weight_relative_error':float(np.linalg.norm(f.V-p.frontend.V)/np.linalg.norm(p.frontend.V)),
        'training_frontend_prediction_max_abs_diff':float(max(abs(corrected-uncorrected)))}
    selected={};selection={}
    for ch in front:
        rows=[r for r in search if r['channel']==ch]
        winner=min(rows,key=lambda r:(r['validation']['ber_upper'],r['validation']['symbol_mse'],r['complexity']['feature_dim'],r['candidate']['name']))
        cc=winner['candidate'];selected[ch]=models[ch][cc['name']];selected[ch].save(out/f'selected_{ch}.npz')
        selection[ch]={'candidate':cc,'model_hash':selected[ch].digest(),'validation':winner['validation'],'complexity':winner['complexity']}
        for name in dict.fromkeys(['original',cc['name']]):
            for seed in [101,102]:
                config=next(r.copy() for r in candidates if r['name']==name);config['seed']=seed
                m=RCModel(config).fit(z[ch]['train'],y[:10000],mask);pred=m.predict(z[ch]['validation'])
                robust.append({'channel':ch,'candidate':name,'seed':seed,'validation':enriched_scores(pred,y[10000:20000],labels[10000:20000],mask)})
    vp={ch:selected[ch].predict(z[ch]['validation']) for ch in front}
    combined=enriched_scores((vp['rx1']+vp['rx2'])/2,y[10000:20000],labels[10000:20000],mask)
    write_json(out/'diagnostics.json',diag);write_json(out/'linear_window_ablation.json',linear)
    write_json(out/'candidate_results.json',search);write_json(out/'seed_stability_validation.json',robust)
    write_json(out/'baseline_train_validation.json',metrics)
    write_json(out/'frozen_selection.json',{'protocol_sha256':file_hash(out/'protocol_before_search.json'),'channels':selection,
        'combined_validation':combined,'xy_weights':[.5,.5],'phase1_config_sha256':file_hash(old/'config.json'),
        'baseline_hashes':{ch:p.frozen_hash() for ch,p in front.items()},'utc':datetime.now(timezone.utc).isoformat()})
    np.savez_compressed(out/'validation_selected_predictions.npz',**vp)
    write_json(out/'search_complete.json',{'test_predictions_computed':False,'candidate_count_per_channel':len(candidates),
        'no_seed_selection':True,'original_baseline_reproduced_on_validation':True})


def confirm(root,out):
    if (out/'confirmation_results.json').exists():raise RuntimeError('Confirmation already evaluated; refusing repeat')
    frozen=json.loads((out/'frozen_selection.json').read_text());assert file_hash(out/'protocol_before_search.json')==frozen['protocol_sha256']
    old=root/'rc_baseline/runs/index509_fixed_v1';c=json.loads((old/'config.json').read_text())
    assert file_hash(old/'config.json')==frozen['phase1_config_sha256']
    a,manifest=load_assets(root,c);start,end=c['splits']['test'];mask=mask_for(end-start,c['guard_symbols'])
    y=(a['i']+1j*a['q'])[start:end];labels=np.column_stack([a['pam_i'],a['pam_q']]).astype(int)[start:end]
    results={};predictions={};timings={};before_hash={}
    for ch in ['rx1','rx2']:
        p=ChannelPipeline.load(old/f'model_{ch}.npz',ch,c);assert p.frozen_hash()==frozen['baseline_hashes'][ch]
        m=RCModel.load(out/f'selected_{ch}.npz',frozen['channels'][ch]['candidate']);assert m.digest()==frozen['channels'][ch]['model_hash']
        before_hash[ch]=m.digest();raw=a[ch][start*4:end*4]
        t=time.perf_counter();r=p.transform(raw,start);shared_time=time.perf_counter()-t
        t=time.perf_counter();selected=m.predict(r['cap_symbols']);infer=time.perf_counter()-t
        predictions[ch]={'old_rc':r['rc'],'selected_rc':selected,'lms':r['lms']}
        results[ch]={k:enriched_scores(v,y,labels,mask) for k,v in predictions[ch].items()}
        timings[ch]={'baseline_pipeline_seconds':shared_time,'selected_rc_only_seconds':infer}
        assert m.digest()==before_hash[ch] and p.frozen_hash()==frozen['baseline_hashes'][ch]
    predictions['xy']={k:(predictions['rx1'][k]+predictions['rx2'][k])/2 for k in ['old_rc','selected_rc','lms']}
    results['xy']={k:enriched_scores(v,y,labels,mask) for k,v in predictions['xy'].items()}
    np.savez_compressed(out/'confirmation_predictions.npz',**{ch+'_'+k:v for ch,d in predictions.items() for k,v in d.items()},mask=mask,symbol_ids=np.arange(start,end))
    write_json(out/'confirmation_results.json',{'evaluation_kind':'previously observed holdout confirmation, not a fresh blind test',
        'frozen_selection_sha256':file_hash(out/'frozen_selection.json'),'metrics':results,'timings':timings,
        'model_hashes_unchanged':True,'test_scored_symbols':int(mask.sum()),'reference_bits':int(mask.sum()*4)})
    plot(out,predictions,y,mask)
    print(json.dumps(results,indent=2),flush=True)


def plot(out,pred,y,mask):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=np.flatnonzero(mask)[::8];ideal=np.unique(y[mask])
    for ch in pred:
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,method in zip(axes,['old_rc','selected_rc','lms']):
            v=pred[ch][method][ids];ax.scatter(v.real,v.imag,s=2,alpha=.22)
            ax.scatter(ideal.real,ideal.imag,c='black',s=15,marker='+')
            ax.set(xlim=(-8,8),ylim=(-8,8),title=method,xlabel='I',ylabel='Q');ax.set_aspect('equal');ax.grid(alpha=.2)
        fig.suptitle(f'509 {ch.upper()} | frozen holdout confirmation');fig.savefig(out/f'constellation_{ch}.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['search','confirm']);parser.add_argument('--out',type=Path,default=Path('runs/phase2_509'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[3]
    (run_search if args.mode=='search' else confirm)(root,args.out.resolve())
if __name__=='__main__':main()
