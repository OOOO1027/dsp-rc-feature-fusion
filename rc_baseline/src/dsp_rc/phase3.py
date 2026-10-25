"""Staged Phase 3: prepare, smoke, search, fit-extra, evaluate.
Extra conditions fit training-only weights with algorithm frozen on509.
"""
import argparse, copy, csv, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .fusion import FusionModel, candidates
from .core import fit_ridge
from .experiment import ChannelPipeline, load_assets, mask_for, write_json, file_hash, check_config
from .phase2 import RCModel, enriched_scores
CHANNELS=['rx1','rx2']
CONTROLS=['tanh80_low','tanh160_low','linear80_low','linear160_low','dual40_low','dual80_low','dual80_states_only']
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def get_config(root,index=509):
    c=read(root/'rc_baseline/runs/index509_fixed_v1/config.json');c['index']=index;return c

def prepare(root,out):
    out.mkdir(parents=True,exist_ok=False)
    metadata=read(root/'audit_outputs/2026-10-16/experiment_metadata.json');rows=[]
    for r in metadata:
        if r['sheet']=='等高线图' and r['cells'].get('12') in [509,513,382]:
            rows.append({'sheet':r['sheet'],'row':r['row'],'conditions':{k:v for k,v in r['cells'].items() if k in [str(i) for i in range(1,9)]+['12']}})
    coverage=list(csv.DictReader((root/'audit_outputs/2026-10-16/index_coverage.csv').open()))
    for idx in [509,513,382]:
        assert any(r['family']=='NLTCP' and int(r['index'])==idx and r['pair']=='True' for r in coverage)
    sources={str(p.relative_to(root)):file_hash(p) for p in (root/'rc_baseline/src/dsp_rc').glob('*.py')};preserved={}
    for folder in ['src','tests','configs','runs/index509_fixed_v1','runs/phase2_509']:
        for p in (root/'rc_baseline'/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in str(p) and 'phase3' not in p.name and p.name!='fusion.py':preserved[str(p.relative_to(root))]=file_hash(p)
    write_json(out/'preserved_before.json',preserved)
    write_json(out/'protocol.json',{'utc':now(),'development_index':509,'extra_indices':[513,382],
        'metadata_selection':rows,'selection_reason':'513 changes VPP200->300 at bias30; 382 changes bias30->50 at VPP200; same rates/rolloff/attenuation. No metadata BER columns or new model BER used.',
        'candidates':candidates(),'selection':'shared RX1/RX2 config per family: min mean(validation upper BER of RX1,RX2,XY); ties mean MSE, feature_dim, name',
        'mandatory_controls':CONTROLS,'final_models':'union of 4 family winners and mandatory controls; no test-based changes',
        'xy_weights':[.5,.5],'seed_policy':'linear200, nonlinear100 fixed; no seed search',
        'new_condition_policy':'Freeze algorithm on509; independently fit frontend, normalization, readouts and LMS on each condition training split; no new-condition tuning.',
        'split_config':get_config(root),'sources_sha256':sources,
        'payload_limit':'Common Tx/reference/labels. 509 test observed historically; extra acquisitions are not independent random payloads.',
        'evaluation_policy':'Freeze all condition weights before any test prediction; one evaluate call, refuse existing evaluation.',
        'standardization':'train-mask window max only; raw states; one passthrough and one bias; all readout coefficients penalized',
        'boundary':'Independent splits, reset states, 128-symbol guards; no cross-split context or truth decoder state.'})
    write_json(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'utc':now()})
    print('Protocol prepared: indices513/382, 18 candidates.',flush=True)

def load_condition(root,out,index,extras=False):
    c=get_config(root,index);a,manifest=load_assets(root,c);dest=out/f'index{index}';dest.mkdir(exist_ok=True);write_json(dest/'input_manifest.json',manifest)
    y=a['i']+1j*a['q'];labels=np.column_stack([a['pam_i'],a['pam_q']]).astype(int);front={};z={};calibration={}
    for ch in CHANNELS:
        if extras:
            t=time.perf_counter();p=ChannelPipeline(ch,c).fit(a[ch][:40000],a['tx'][:40000],y[:10000]);secs=time.perf_counter()-t;p.save(dest/f'frontend_{ch}.npz')
            calibration[ch]={'fit_seconds':secs,'alignment':p.alignment,'train_symbol_correlations':p.symbol_alignment_diagnostic,
                'best_training_symbol_lag':int(max(p.symbol_alignment_diagnostic,key=p.symbol_alignment_diagnostic.get))}
            if calibration[ch]['best_training_symbol_lag']!=0:raise RuntimeError(f'Condition{index} needs train-only alignment investigation')
        else:p=ChannelPipeline.load(root/f'rc_baseline/runs/index509_fixed_v1/model_{ch}.npz',ch,c)
        front[ch]=p;z[ch]={}
        for split in (['train'] if extras else ['train','validation']):
            start,end=c['splits'][split];z[ch][split]=p.preprocess(a[ch][4*start:4*end],start)[1]
    if extras:write_json(dest/'training_calibration.json',calibration)
    return c,a,y,labels,front,z

def smoke(root,out):
    assert (out/'protocol.json').exists() and not (out/'smoke.json').exists()
    c=get_config(root);a,_=load_assets(root,c);y=a['i']+1j*a['q'];labels=np.column_stack([a['pam_i'],a['pam_q']]).astype(int)
    result={};config=next(c for c in candidates() if c['name']=='dual40_low')
    for ch in CHANNELS:
        p=ChannelPipeline(ch,c).fit(a[ch][:12000],a['tx'][:12000],y[:3000])
        tr=p.preprocess(a[ch][:12000],0)[1];va=p.preprocess(a[ch][12000:16000],3000)[1]
        m=FusionModel(config).fit(tr,y[:3000],mask_for(3000,128));path=out/f'smoke_{ch}.npz';m.save(path)
        pred=m.predict(va);np.testing.assert_array_equal(pred,FusionModel.load(path).predict(va))
        result[ch]={'metrics':enriched_scores(pred,y[3000:4000],labels[3000:4000],mask_for(1000,128)),
            'finite':bool(np.isfinite(pred).all()),'save_load_exact':True,'stats':m.stats}
    write_json(out/'smoke.json',{'training_symbols':[0,3000],'check_symbols':[3000,4000],'results':result});print('Small end-to-end passed.',flush=True)

def key(row):
    v=row['validation'];return (sum(v[ch]['ber_upper'] for ch in ['rx1','rx2','xy'])/3,
        sum(v[ch]['symbol_mse'] for ch in ['rx1','rx2','xy'])/3,row['complexity']['rx1']['feature_dim'],row['config']['name'])

def feature_diagnostics(model,ztrain,zval,mask):
    tr=model.extract(ztrain);va=model.extract(zval);xtr=np.column_stack([tr['input'],np.ones(len(ztrain))]);xva=np.column_stack([va['input'],np.ones(len(zval))]);result={}
    for name,states in tr['states'].items():
        w=fit_ridge(xtr[mask],states[mask],1e-6);target=va['states'][name][mask];error=target-xva[mask]@w
        result[name+'_variance_unexplained_by_current_window']=float(np.sum(error**2)/np.sum((target-states[mask].mean(axis=0))**2))
    if 'linear' in tr['states'] and 'nonlinear' in tr['states']:
        ftr=np.column_stack([xtr,tr['states']['linear']]);fva=np.column_stack([xva,va['states']['linear']]);w=fit_ridge(ftr[mask],tr['states']['nonlinear'][mask],1e-6)
        target=va['states']['nonlinear'][mask]
        result['nonlinear_variance_unexplained_by_window_and_linear']=float(np.sum((target-fva[mask]@w)**2)/np.sum((target-tr['states']['nonlinear'][mask].mean(axis=0))**2))
    s=np.linalg.svd(tr['features'][mask],compute_uv=False)
    result.update(feature_dim=len(s),numerical_rank_rtol_1e_10=int(np.sum(s>s[0]*1e-10)),singular_max=float(s[0]),singular_min=float(s[-1]),
                  gram_condition_regularized=float((s[0]**2+model.config['ridge'])/(s[-1]**2+model.config['ridge'])))
    return result

def search(root,out):
    assert not (out/'frozen_algorithm.json').exists() and (out/'smoke.json').exists()
    protocol=read(out/'protocol.json');c,a,y,labels,front,z=load_condition(root,out,509);mask=mask_for(10000,128)
    rows=[];models={};vp={};trainp={};parity={}
    for config in protocol['candidates']:
        cc=copy.deepcopy(c);cc['rc']['window_symbols']=config['taps'];check_config(cc)
        name=config['name'];models[name]={};vp[name]={};trainp[name]={};complexity={};times={}
        for ch in CHANNELS:
            m=FusionModel(config).fit(z[ch]['train'],y[:10000],mask);digest=m.digest();predtr=m.predict(z[ch]['train'])
            t=time.perf_counter();pred=m.predict(z[ch]['validation']);secs=time.perf_counter()-t
            path=out/f'index509/{name}_{ch}.npz';m.save(path);loaded=FusionModel.load(path)
            np.testing.assert_array_equal(pred,loaded.predict(z[ch]['validation']));assert digest==m.digest()==loaded.digest()
            models[name][ch]=m;vp[name][ch]=pred;trainp[name][ch]=predtr;times[ch]=secs;complexity[ch]=m.stats
            if name=='tanh80_low':
                frozen=read(root/'rc_baseline/runs/phase2_509/frozen_selection.json')
                old=RCModel.load(root/f'rc_baseline/runs/phase2_509/selected_{ch}.npz',frozen['channels'][ch]['candidate'])
                np.testing.assert_array_equal(pred,old.predict(z[ch]['validation']));parity[ch]=True
        for p in [vp[name],trainp[name]]:p['xy']=(p['rx1']+p['rx2'])/2
        row={'config':config,'validation':{ch:enriched_scores(p,y[10000:20000],labels[10000:20000],mask) for ch,p in vp[name].items()},
            'train':{ch:enriched_scores(p,y[:10000],labels[:10000],mask) for ch,p in trainp[name].items()},'complexity':complexity,'validation_inference_seconds':times}
        rows.append(row);print(name,'validation mean upper',key(row)[0],flush=True)
    winners={family:min([r for r in rows if r['config']['family']==family],key=key)['config']['name'] for family in ['A','B','C','D']}
    final=list(dict.fromkeys(list(winners.values())+protocol['mandatory_controls']));write_json(out/'validation_results.json',rows)
    np.savez_compressed(out/'validation_predictions.npz',**{name+'_'+ch:p for name,v in vp.items() for ch,p in v.items()},mask=mask,symbol_ids=np.arange(10000,20000))
    diagnostics={name:{ch:feature_diagnostics(models[name][ch],z[ch]['train'],z[ch]['validation'],mask) for ch in CHANNELS} for name in dict.fromkeys(['dual40_low','dual80_low',winners['D']])}
    write_json(out/'feature_diagnostics_validation.json',diagnostics)
    write_json(out/'frozen_algorithm.json',{'utc':now(),'protocol_sha256':file_hash(out/'protocol.json'),'family_winners':winners,
        'evaluated_names':final,'selected_configs':{name:models[name]['rx1'].config for name in final},'phase2_validation_exact':parity,
        'test_predictions_computed':False,'models509':{name:{ch:models[name][ch].digest() for ch in CHANNELS} for name in final}})
    print('Frozen family winners:',winners,'Final models:',final,flush=True)

def fit_extra(root,out):
    assert not (out/'frozen_all_models.json').exists();frozen=read(out/'frozen_algorithm.json');protocol=read(out/'protocol.json')
    assert frozen['protocol_sha256']==file_hash(out/'protocol.json');hashes={};frontend_hashes={};times={}
    for index in [509]+protocol['extra_indices']:
        if index==509:
            hashes[str(index)]=frozen['models509'];c=get_config(root);front={ch:ChannelPipeline.load(root/f'rc_baseline/runs/index509_fixed_v1/model_{ch}.npz',ch,c) for ch in CHANNELS}
        else:
            c,a,y,labels,front,z=load_condition(root,out,index,extras=True);hashes[str(index)]={};times[str(index)]={}
            for name,config in frozen['selected_configs'].items():
                hashes[str(index)][name]={};times[str(index)][name]={}
                for ch in CHANNELS:
                    m=FusionModel(config).fit(z[ch]['train'],y[:10000],mask_for(10000,128));m.save(out/f'index{index}/{name}_{ch}.npz')
                    hashes[str(index)][name][ch]=m.digest();times[str(index)][name][ch]=m.stats
            print('All models fitted on training only:',index,flush=True)
        frontend_hashes[str(index)]={ch:p.frozen_hash() for ch,p in front.items()}
    manifests=[read(out/f'index{idx}/input_manifest.json') for idx in [509]+protocol['extra_indices']]
    same={k:len({m[k]['sha256'] for m in manifests})==1 for k in ['tx','i','q','pam_i','pam_q']}
    write_json(out/'training_extra.json',times)
    write_json(out/'frozen_all_models.json',{'utc':now(),'algorithm_sha256':file_hash(out/'frozen_algorithm.json'),'model_hashes':hashes,
        'frontend_hashes':frontend_hashes,'test_predictions_computed':False,'shared_reference_hashes_identical':same,
        'sources_sha256':{str(p.relative_to(root)):file_hash(p) for p in (root/'rc_baseline/src/dsp_rc').glob('*.py')}})
    print('All condition weights frozen BEFORE any test prediction.',flush=True)

def evaluate(root,out):
    assert not (out/'evaluation_started.json').exists(),'Evaluation already started; refusing repeat.'
    frozen=read(out/'frozen_all_models.json');algo=read(out/'frozen_algorithm.json');protocol=read(out/'protocol.json')
    assert frozen['algorithm_sha256']==file_hash(out/'frozen_algorithm.json') and algo['protocol_sha256']==file_hash(out/'protocol.json')
    for path,h in frozen['sources_sha256'].items():assert file_hash(root/path)==h
    write_json(out/'evaluation_started.json',{'utc':now(),'frozen_sha256':file_hash(out/'frozen_all_models.json')});results={}
    for index in [509]+protocol['extra_indices']:
        c=get_config(root,index);a,_=load_assets(root,c);start,end=c['splits']['test'];mask=mask_for(end-start,128)
        y=(a['i']+1j*a['q'])[start:end];labels=np.column_stack([a['pam_i'],a['pam_q']]).astype(int)[start:end];pred={};timings={};preseconds={};metrics={}
        for ch in CHANNELS:
            fp=root/f'rc_baseline/runs/index509_fixed_v1/model_{ch}.npz' if index==509 else out/f'index{index}/frontend_{ch}.npz'
            p=ChannelPipeline.load(fp,ch,c);assert p.frozen_hash()==frozen['frontend_hashes'][str(index)][ch]
            t=time.perf_counter();z=p.preprocess(a[ch][4*start:4*end],start)[1];preseconds[ch]=time.perf_counter()-t
            pred[ch]={'before':z*p.before_gain};timings[ch]={};t=time.perf_counter();pred[ch]['lms']=p.lms.predict(z)*p.target_scale;timings[ch]['lms']=time.perf_counter()-t
            for name in algo['evaluated_names']:
                m=FusionModel.load(out/f'index{index}/{name}_{ch}.npz');h=frozen['model_hashes'][str(index)][name][ch];assert m.digest()==h
                t=time.perf_counter();pred[ch][name]=m.predict(z);timings[ch][name]=time.perf_counter()-t;assert m.digest()==h
            assert p.frozen_hash()==frozen['frontend_hashes'][str(index)][ch]
        pred['xy']={name:(pred['rx1'][name]+pred['rx2'][name])/2 for name in pred['rx1']}
        for ch in pred:metrics[ch]={name:enriched_scores(v,y,labels,mask) for name,v in pred[ch].items()}
        dest=out/f'index{index}';np.savez_compressed(dest/'test_predictions.npz',**{ch+'_'+name:p for ch,v in pred.items() for name,p in v.items()},mask=mask,symbol_ids=np.arange(start,end))
        results[str(index)]={'metrics':metrics,'inference_seconds':timings,'preprocess_seconds':preseconds,'scored_symbols':int(mask.sum()),'reference_bits':int(4*mask.sum()),'hashes_unchanged':True}
        write_json(dest/'test_results.json',results[str(index)]);print(index,'frozen evaluation complete',flush=True)
    write_json(out/'test_results.json',{'utc':now(),'conditions':results,'evaluation_count':1,'509_kind':'historically observed holdout regression',
        'new_conditions_kind':'preselected condition-specific trained holdouts, common payload; not unseen-message generalization'})
    previous=read(out/'preserved_before.json');assert all(file_hash(root/p)==h for p,h in previous.items())
    with np.load(out/'index509/test_predictions.npz') as new,np.load(root/'rc_baseline/runs/phase2_509/confirmation_predictions.npz') as old:
        np.testing.assert_array_equal(new['mask'],old['mask'])
        for ch in ['rx1','rx2','xy']:
            np.testing.assert_array_equal(new[ch+'_tanh80_low'],old[ch+'_selected_rc']);np.testing.assert_array_equal(new[ch+'_lms'],old[ch+'_lms'])
    write_json(out/'verification.json',{'old_files_unchanged':True,'preserved_file_count':len(previous),'phase2_validation_exact':algo['phase2_validation_exact'],
        'phase2_test_predictions_exact':True,'same_test_mask':True,'all_model_hashes_unchanged':True,'one_final_evaluation':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','search','fit-extra','evaluate']);p.add_argument('--out',type=Path,default=Path('runs/phase3_feature_fusion'))
    args=p.parse_args();root=Path(__file__).resolve().parents[3]
    {'prepare':prepare,'smoke':smoke,'search':search,'fit-extra':fit_extra,'evaluate':evaluate}[args.mode](root,args.out.resolve())
if __name__=='__main__':main()
