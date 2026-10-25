from dsp_rc.portable import load_audit_index
"""Residual diagnostics and Tx provenance; no model selection or test prediction."""
import argparse,csv
from pathlib import Path
import numpy as np
from .phase4 import data
from .residual_fusion import FeatureBank,reference_config
from .fusion import FusionModel
from .experiment import mask_for,write_json,file_hash

def describe(error,z,y,h,mask,bins):
 e=error[mask];zz=z[mask];yy=y[mask];state=h[mask];result={}
 for k,v in [('I',e.real),('Q',e.imag)]:
  centered=v-v.mean();variance=np.mean(centered**2)
  result[k]={'mean':float(v.mean()),'variance':float(variance),'quantiles':np.quantile(v,[0,.01,.05,.5,.95,.99,1]).tolist(),
      'skewness':float(np.mean(centered**3)/variance**1.5),'kurtosis':float(np.mean(centered**4)/variance**2),
      'acf_lags_0_to64':[1.]+[float(np.dot(centered[:-lag],centered[lag:])/np.dot(centered,centered)) for lag in range(1,65)]}
 residualpower=abs(e)**2;amplitude=abs(zz);groups=np.digitize(amplitude,bins)
 result['receive_amplitude_bins_train_thresholds']=bins.tolist()
 result['receive_amplitude_quintiles']=[{'bin':i,'count':int(np.sum(groups==i)),'mean_rx_amplitude':float(amplitude[groups==i].mean()),'residual_mse':float(residualpower[groups==i].mean())} for i in range(5)]
 result['corr_receive_amplitude_residual_abs']=float(np.corrcoef(amplitude,abs(e))[0,1])
 result['constellation_regions']=[]
 for re in [-5,-3,-1,1,3,5]:
  for im in [-5,-3,-1,1,3,5]:
   sel=(yy.real==re)&(yy.imag==im)
   result['constellation_regions'].append({'I':re,'Q':im,'count':int(sel.sum()),'mse':float(residualpower[sel].mean()),'error_energy_fraction':float(residualpower[sel].sum()/residualpower.sum())})
 high=(abs(yy.real)==5)|(abs(yy.imag)==5)
 result['outer_level_region']={'symbol_fraction':float(high.mean()),'error_energy_fraction':float(residualpower[high].sum()/residualpower.sum()),'mse_outer':float(residualpower[high].mean()),'mse_inner':float(residualpower[~high].mean())}
 s=state-state.mean(axis=0);target=np.column_stack([e.real,e.imag]);target-=target.mean(axis=0)
 cor=s.T@target/np.sqrt((s*s).sum(axis=0)[:,None]*(target*target).sum(axis=0)[None,:])
 result['nonlinear_state_correlations']=cor.tolist();result['max_abs_state_corr_per_IQ']=np.max(abs(cor),axis=0).tolist()
 return result

def run(root,out):
 assert not (out/'residual_diagnostics.json').exists();z,y,labels,manifest=data(root);mask=mask_for(10000,128);result={};arrays={}
 for ch in ['rx1','rx2']:
  bank=FeatureBank(100).fit(z[ch]['train'],mask);_,ht=bank.transform(z[ch]['train']);_,hv=bank.transform(z[ch]['validation']);bins=np.quantile(abs(z[ch]['train'][mask]),[.2,.4,.6,.8]);result[ch]={}
  for base in ['ridge_low','linear80_low']:
   m=FusionModel(reference_config(base)).fit(z[ch]['train'],y[:10000],mask);result[ch][base]={}
   for split,start,h in [('train',0,ht),('validation',10000,hv)]:
    error=y[start:start+10000]-m.predict(z[ch][split]);result[ch][base][split]=describe(error,z[ch][split],y[start:start+10000],h,mask,bins)
    arrays[f'{ch}_{base}_{split}_error']=error
   arrays[f'{ch}_{base}_train_mse']=np.array(np.mean(abs(arrays[f'{ch}_{base}_train_error'][mask])**2))
  for split in ['train','validation']:arrays[ch+'_'+split+'_input']=z[ch][split]
 arrays['mask']=mask;arrays['ideal_validation']=y[10000:];np.savez_compressed(out/'residual_diagnostics_arrays.npz',**arrays);write_json(out/'residual_diagnostics.json',result)
 # Distinct waveform bytes do not imply independent payloads.
 nroot=root/'NN数据20250320/NN数据20250320';txrows=[];reference=np.loadtxt(nroot/'txdata/RF0.105_data36QAM.txt').ravel()
 audit=load_audit_index(root)
 for p in sorted((nroot/'txdata').glob('*')):
  h=file_hash(p);assert h==audit[str(p)]['sha256'];row={'path':str(p),'bytes':p.stat().st_size,'sha256':h,'kind':p.suffix}
  if p.suffix=='.txt':
   a=np.loadtxt(p);row['shape']=list(a.shape)
   if a.size==reference.size and a.ndim==1:
    row['identical_to_RF0105']=bool(np.array_equal(a,reference));row['train_prefix_correlation_RF0105']=float(np.corrcoef(a[:40000],reference[:40000])[0,1])
  txrows.append(row)
 metadata=__import__('json').loads((root/'audit_outputs/2026-10-16/experiment_metadata.json').read_text());txmentions=[r for r in metadata if r['row']==2]
 labels_found=[str(p) for p in nroot.rglob('*.txt') if p.name.startswith('decimal_') or p.name in ['v1.txt','v2.txt']]
 write_json(out/'payload_inventory.json',{'tx_files':txrows,'metadata_Tx_rows':txmentions,'label_symbol_files':labels_found,
   'conclusion':'Several different waveform files exist; no separately identified NLTCP payload with complete symbols/PAM4 labels and matching RX metadata verified. Other waveform names alone do not establish modulation, payload identity or compatible NLTCP labels. Do not infer fresh payload from differing waveform hashes.'})
 print('Residual diagnostics and Tx inventory saved; no test predictions.',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('runs/phase4_residual_fusion'));a=p.parse_args();run(Path(__file__).resolve().parents[3],a.out.resolve())
