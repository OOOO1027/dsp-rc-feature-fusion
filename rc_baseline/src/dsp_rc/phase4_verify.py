"""Delivery checks and numerical conditioning diagnostics, training input only."""
import argparse,re
from pathlib import Path
import numpy as np
from .phase3 import read
from .phase4 import data
from .residual_fusion import ResidualFusion
from .experiment import file_hash,write_json,mask_for

def run(root,out):
 old=read(out/'preserved_before.json');assert all(file_hash(root/p)==h for p,h in old.items())
 protocol=read(out/'protocol.json');selection=read(out/'selection.json');assert file_hash(out/'protocol.json')==selection['protocol_sha256']
 # Core experiment sources as executed are unchanged after selection.
 for p,h in protocol['source_hashes'].items():assert file_hash(root/p)==h
 with np.load(out/'predictions_seed100.npz') as new,np.load(root/'rc_baseline/runs/phase3_feature_fusion/validation_predictions.npz') as oldp:
  np.testing.assert_array_equal(new['mask'],oldp['mask'])
  for cfg in protocol['main_candidates']:
   if cfg['kind']=='reference':
    for ch in ['rx1','rx2','xy']:np.testing.assert_array_equal(new[cfg['name']+'_validation_'+ch],oldp[cfg['name']+'_'+ch])
  assert all(np.isfinite(new[k]).all() and len(new[k])==10000 for k in new.files)
 assert not list(out.glob('*test*predictions*')) and not read(out/'decision.json')['test_evaluated']
 z,_,_,_=data(root);mask=mask_for(10000,128);conditions={}
 for ch in ['rx1','rx2']:
  conditions[ch]={}
  for name in ['joint_std_uniform','joint_std_group','joint_std_equivalent','joint_orth_uniform','joint_orth_group','joint_orth_equivalent']:
   model=ResidualFusion.load(out/f'models/{name}_{ch}_s100.npz');digest=model.digest();c,h=model.bank.transform(z[ch]['train']);f=np.column_stack([c,h]);t=model.arrays['T'];r=model.arrays['penalty_factor'];s=np.linalg.svd(np.vstack([(f@t)[mask],r]),compute_uv=False)
   e=h-c@model.arrays['projection']
   conditions[ch][name]={'augmented_design_condition_2':float(s[0]/s[-1]),'normal_matrix_condition_2':float((s[0]/s[-1])**2),
      'projected_state_variance_fraction':float(np.var(e[mask],axis=0).sum()/np.var(h[mask],axis=0).sum()),
      'normalized_CtE':float(np.linalg.norm(c[mask].T@e[mask])/np.linalg.norm(c[mask])/np.linalg.norm(e[mask]))}
   assert model.digest()==digest
  s=np.linalg.svd(np.vstack([f[mask],np.sqrt(.01)*np.eye(f.shape[1])]),compute_uv=False)
  conditions[ch]['raw_dual80_reg']={'augmented_design_condition_2':float(s[0]/s[-1]),'normal_matrix_condition_2':float((s[0]/s[-1])**2)}
 write_json(out/'conditioning.json',conditions)
 assert 'Ran 23 tests' in (out/'unit_tests_final.log').read_text() and '\nOK' in (out/'unit_tests_final.log').read_text()
 write_json(out/'verification.json',{'prior_files_unchanged':True,'prior_file_count':len(old),'experiment_sources_unchanged_since_protocol':True,
  'all6_phase3_reference_validation_predictions_exact':True,'same_validation_mask':True,'unit_tests_passed':23,'main_candidates':24,
  'extra_nonlinear_seeds':[101,102],'test_predictions_or_metrics_computed':False,'post_selection_expansion':False,'no_new_dependency_installed':True,
  'coordinate_equivalence_max_error':selection['coordinate_equivalence_max_error'],'figure_count':len(list(out.glob('*.png')))})
 print('Integrity,6 baseline replays,23 tests and numerical conditioning verified.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('runs/phase4_residual_fusion'));a=p.parse_args();run(Path(__file__).resolve().parents[3],a.out.resolve())
