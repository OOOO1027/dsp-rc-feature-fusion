"""Validation-only Phase4 runner. No test prediction code or test mode."""
import argparse,json,time,sys,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .core import decode_nltcp
from .fusion import FusionModel
from .residual_fusion import configurations,reference_config,FeatureBank,ResidualFusion
from .phase3 import get_config,read
from .experiment import load_assets,ChannelPipeline,mask_for,write_json,file_hash
from .phase2 import enriched_scores
CH=['rx1','rx2'];OUTPUTS=CH+['xy']
def now():return datetime.now(timezone.utc).isoformat()
def aggregate(metrics,k):return float(np.mean([metrics[ch][k] for ch in OUTPUTS]))
def prepare(root,out):
 out.mkdir(parents=True,exist_ok=False);preserved={}
 for folder in ['src','tests','configs','runs/index509_fixed_v1','runs/phase2_509','runs/phase3_feature_fusion']:
  for p in (root/'rc_baseline'/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in str(p) and 'phase4' not in p.name and p.name!='residual_fusion.py':preserved[str(p.relative_to(root))]=file_hash(p)
 write_json(out/'preserved_before.json',preserved)
 write_json(out/'protocol.json',{'utc':now(),'index':509,'split_config':get_config(root),'used_splits':['train','validation'],
 'test_policy':'No test predictions, scores or choices; historical509/513/382 test rows not used by runner.',
 'main_candidates':configurations(),'eta_grid':[0,.25,.5,1.],'eta_zero':'Four family eta0 outputs alias their existing linear baselines; no extra fitted candidate.',
 'projection_ridge':.01,'residual_ridge':.01,'group_penalties':{'input':.01,'linear':1.,'nonlinear':10.,'bias':.01},
 'selection':'mean validation complex MSE of RX1,RX2,XY; tie mean upper BER, smaller eta, name. Shared config/eta across channels; eta0 may win.',
 'seed_checks':'Check best positive-eta residual and best non-equivalent joint at nonlinear seeds100,101,102, linear200 fixed. No seed retuning, no candidate expansion.',
 'stop_rule':'For each shortlisted method require all3seeds lower mean MSE versus strongest linear MSE baseline, >=0.1% mean MSE gain, no seed worse upperBER versus strongest linear upperBER baseline, and paired bootstrap95% upper delta<0 for both metrics at both block256/512. Otherwise stop without expansion.',
 'bootstrap':{'samples':2000,'seed':404,'block_lengths':[256,512],'method':'paired moving blocks of predecoded per-symbol losses; shared indices across models/channels/seeds; never re-decode concatenated blocks'},
 'diagnostics':'training/validation IQ distribution, ACF1..64, receive amplitude train-quintiles, ideal constellation residual energy, state correlations and heldout residual predictability',
 'new_conditions':'No extra-condition fits this round. Reused validation/post-selection intervals are exploratory.',
 'source_hashes':{str(p.relative_to(root)):file_hash(p) for p in (root/'rc_baseline/src/dsp_rc').glob('*.py')}})
 write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__});print('Fixed24 candidates; eta0 aliases; 3seeds; no test mode.',flush=True)
def data(root):
 c=get_config(root);a,manifest=load_assets(root,c)
 # Existing loader checks whole-file integrity; discard all test rows immediately.
 for k in a:a[k]=a[k][:80000 if k in ['rx1','rx2','tx'] else 20000].copy()
 y=a['i']+1j*a['q'];labels=np.column_stack([a['pam_i'],a['pam_q']]).astype(int);z={}
 for ch in CH:
  p=ChannelPipeline.load(root/f'rc_baseline/runs/index509_fixed_v1/model_{ch}.npz',ch,c);z[ch]={}
  for split,(start,end) in [('train',(0,10000)),('validation',(10000,20000))]:z[ch][split]=p.preprocess(a[ch][4*start:4*end],start)[1]
 return z,y,labels,manifest

def fit_model(cfg,z,y,mask,seed,bank=None):
 if cfg['kind']=='reference':
  m=FusionModel(reference_config(cfg['reference'],seed)).fit(z,y,mask);m.stats['readout_parameters']=m.stats['trained_parameters'];m.stats['readout_dimension']=m.stats['feature_dim'];m.stats['projection_fitted_coefficients']=0;return m
 return ResidualFusion(cfg,seed).fit(z,y,mask,bank)
def smoke(root,out):
 assert not (out/'smoke.json').exists();z,y,labels,manifest=data(root);cfg=next(c for c in configurations() if c['kind']=='residual' and c['orth']);rows={}
 for ch in CH:
  m=fit_model(cfg,z[ch]['train'][:3000],y[:3000],mask_for(3000,128),100);pred=m.predict(z[ch]['train'][3000:4000]);assert np.isfinite(pred).all();path=out/f'smoke_{ch}.npz';m.save(path)
  np.testing.assert_array_equal(pred,ResidualFusion.load(path).predict(z[ch]['train'][3000:4000]));rows[ch]=True
 write_json(out/'smoke.json',{'passed':rows,'scope':'interface only on original train0:4000; front frozen on original full train, not an independent small holdout'});print('Train-only interface smoke passed.',flush=True)
def run_candidates(root,out):
 assert (out/'smoke.json').exists() and not (out/'selection.json').exists()
 protocol=read(out/'protocol.json');z,y,labels,manifest=data(root);write_json(out/'input_manifest.json',manifest);mask=mask_for(10000,128);modelsdir=out/'models';modelsdir.mkdir();rows=[];allpred={};banks={ch:FeatureBank(100).fit(z[ch]['train'],mask) for ch in CH}
 for cfg in protocol['main_candidates']:
  name=cfg['name'];pred={s:{} for s in ['train','validation']};stats={};times={}
  for ch in CH:
   m=fit_model(cfg,z[ch]['train'],y[:10000],mask,100,banks[ch]);digest=m.digest();path=modelsdir/f'{name}_{ch}_s100.npz';m.save(path)
   for split in pred:
    t=time.perf_counter();pred[split][ch]=m.predict(z[ch][split]);times[ch+'_'+split]=time.perf_counter()-t
   loaded=FusionModel.load(path) if cfg['kind']=='reference' else ResidualFusion.load(path);np.testing.assert_array_equal(pred['validation'][ch],loaded.predict(z[ch]['validation']));assert m.digest()==digest;stats[ch]=m.stats
  for split in pred:pred[split]['xy']=(pred[split]['rx1']+pred[split]['rx2'])/2
  row={'config':cfg,'stats':stats,'inference_seconds':times}
  for split,start in [('train',0),('validation',10000)]:
   row[split]={ch:enriched_scores(v,y[start:start+10000],labels[start:start+10000],mask) for ch,v in pred[split].items()}
   for ch,v in pred[split].items():allpred[f'{name}_{split}_{ch}']=v
  rows.append(row);print(name,'MSE',aggregate(row['validation'],'symbol_mse'),'upper',aggregate(row['validation'],'ber_upper'),flush=True)
 references={r['config']['name']:r for r in rows if r['config']['kind']=='reference'};linear=[references[k] for k in ['ridge_low','linear80_low','linear160_low']]
 key=lambda r:(aggregate(r['validation'],'symbol_mse'),aggregate(r['validation'],'ber_upper'),r['config']['eta'] or 0,r['config']['name'])
 best_res=min([r for r in rows if r['config']['kind']=='residual'],key=key);best_joint=min([r for r in rows if r['config']['kind']=='joint' and r['config']['penalty_mode']!='equivalent'],key=key)
 best_linear_mse=min(linear,key=key)['config']['name'];best_linear_upper=min(linear,key=lambda r:aggregate(r['validation'],'ber_upper'))['config']['name'];families={}
 for base in ['ridge_low','linear80_low']:
  for orth in [False,True]:
   options=[r for r in rows if r['config']['kind']=='residual' and r['config']['base']==base and r['config']['orth']==orth]+[references[base]];families[base+('_orth' if orth else '_raw')]=min(options,key=key)['config']['name']
 equivalence={}
 for name in ['joint_std_equivalent','joint_orth_equivalent']:
  equivalence[name]={ch:float(np.max(abs(allpred[f'{name}_validation_{ch}'][mask]-allpred[f'dual80_reg_validation_{ch}'][mask]))) for ch in CH};assert max(equivalence[name].values())<1e-6,equivalence
 write_json(out/'candidate_results.json',rows);np.savez_compressed(out/'predictions_seed100.npz',**allpred,mask=mask)
 write_json(out/'selection.json',{'utc':now(),'protocol_sha256':file_hash(out/'protocol.json'),'residual_family_winners_including_eta0':families,'shortlisted_positive_residual':best_res['config'],'shortlisted_joint':best_joint['config'],'strongest_linear_mse':best_linear_mse,'strongest_linear_upper':best_linear_upper,'coordinate_equivalence_max_error':equivalence,'test_evaluated':False,'selection_is_validation_exploratory':True})
 print('Frozen shortlist',best_res['config']['name'],best_joint['config']['name'],'eta0-aware winners',families,flush=True)
def per_symbol(pred,y,labels,mask):
 _,d=decode_nltcp(pred);d=d[mask];lab=labels[mask];valid=d<=3;pop=np.array([int(i).bit_count() for i in range(8)]);errors=pop[np.bitwise_xor(d,lab)];upper=np.where(valid,errors,2).sum(axis=1)/4
 return np.column_stack([abs(pred[mask]-y[mask])**2,upper])
def paired_blocks(delta,length,reps=2000,seed=404):
 rng=np.random.default_rng(seed);n=len(delta);nblock=(n+length-1)//length;tail=n-(nblock-1)*length;cumulative=np.vstack([np.zeros((1,delta.shape[1])),np.cumsum(delta,axis=0)])
 starts=rng.integers(0,n-length+1,size=(reps,nblock));ends=starts+length;ends[:,-1]=starts[:,-1]+tail;means=np.sum(cumulative[ends]-cumulative[starts],axis=1)/n
 return {'mean_delta':delta.mean(axis=0).tolist(),'CI95':np.quantile(means,[.025,.975],axis=0).T.tolist(),'block_length':length,'replicates':reps}
def seeds(root,out):
 assert not (out/'decision.json').exists();selection=read(out/'selection.json');assert selection['protocol_sha256']==file_hash(out/'protocol.json')
 z,y,labels,_=data(root);mask=mask_for(10000,128);rows=[];preds={};saved=np.load(out/'predictions_seed100.npz');configs=[selection['shortlisted_positive_residual'],selection['shortlisted_joint']]
 for cfg in configs:
  name=cfg['name']
  for seed in [100,101,102]:
   p={};stats={}
   for ch in CH:
    if seed==100:p[ch]=saved[f'{name}_validation_{ch}'];continue
    m=fit_model(cfg,z[ch]['train'],y[:10000],mask,seed);m.save(out/f'models/{name}_{ch}_s{seed}.npz');digest=m.digest();t=time.perf_counter();p[ch]=m.predict(z[ch]['validation']);secs=time.perf_counter()-t;assert digest==m.digest();stats[ch]=dict(m.stats,validation_inference_seconds=secs)
    np.testing.assert_array_equal(p[ch],ResidualFusion.load(out/f'models/{name}_{ch}_s{seed}.npz').predict(z[ch]['validation']))
   p['xy']=(p['rx1']+p['rx2'])/2;metrics={ch:enriched_scores(v,y[10000:],labels[10000:],mask) for ch,v in p.items()};rows.append({'name':name,'seed':seed,'validation':metrics,'stats':stats})
   for ch in p:preds[f'{name}_{ch}_s{seed}']=p[ch]
   print(name,'seed',seed,'MSE',aggregate(metrics,'symbol_mse'),'upper',aggregate(metrics,'ber_upper'),flush=True)
 baselines={metric:selection[key] for metric,key in [('mse','strongest_linear_mse'),('upper','strongest_linear_upper')]};loss_base={}
 for metric,name in baselines.items():loss_base[metric]=np.mean([per_symbol(saved[f'{name}_validation_{ch}'],y[10000:],labels[10000:],mask) for ch in OUTPUTS],axis=0)
 decisions={};boot={}
 for cfg in configs:
  name=cfg['name'];deltas=[];seed_deltas=[]
  for seed in [100,101,102]:
   losses=np.mean([per_symbol(preds[f'{name}_{ch}_s{seed}'],y[10000:],labels[10000:],mask) for ch in OUTPUTS],axis=0);delta=losses-np.column_stack([loss_base['mse'][:,0],loss_base['upper'][:,1]]);deltas.append(delta);seed_deltas.append(delta.mean(axis=0))
  avg=np.mean(deltas,axis=0);boot[name]={str(b):paired_blocks(avg,b) for b in [256,512]};base_mse=float(loss_base['mse'][:,0].mean());gain=-float(avg[:,0].mean())/base_mse
  passed=all(d[0]<0 and d[1]<=0 for d in seed_deltas) and gain>=.001 and all(np.all(np.array(b['CI95'])[:,1]<0) for b in boot[name].values());decisions[name]={'pass_gate':bool(passed),'relative_mean_mse_gain':gain,'seed_mean_deltas':np.asarray(seed_deltas).tolist()}
 write_json(out/'seed_results.json',rows);np.savez_compressed(out/'seed_validation_predictions.npz',**preds,mask=mask);write_json(out/'paired_block_bootstrap.json',boot)
 write_json(out/'decision.json',{'utc':now(),'methods':decisions,'continue_to_new_payload_preparation':any(d['pass_gate'] for d in decisions.values()),'test_evaluated':False,'further_search_performed':False,'inference_limit':'Post-selection exploratory block intervals on reused validation; not independent confidence after model selection.'});print('Gate:',decisions,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','search','seeds']);p.add_argument('--out',type=Path,default=Path('runs/phase4_residual_fusion'));a=p.parse_args();root=Path(__file__).resolve().parents[3];{'prepare':prepare,'smoke':smoke,'search':run_candidates,'seeds':seeds}[a.mode](root,a.out.resolve())
if __name__=='__main__':main()
