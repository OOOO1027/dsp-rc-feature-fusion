"""Independent SVD check of frozen readouts, training/validation data only."""
import argparse
from pathlib import Path
import numpy as np
from .phase3 import read,get_config
from .experiment import load_assets,ChannelPipeline,mask_for,write_json
from .fusion import FusionModel

def run(root,out):
    frozen=read(out/'frozen_algorithm.json');c=get_config(root);a,_=load_assets(root,c)
    target=np.column_stack([a['i'][:10000],a['q'][:10000]]);mask=mask_for(10000,128);results={}
    for ch in ['rx1','rx2']:
        p=ChannelPipeline.load(root/f'rc_baseline/runs/index509_fixed_v1/model_{ch}.npz',ch,c)
        tr=p.preprocess(a[ch][:40000],0)[1];va=p.preprocess(a[ch][40000:80000],10000)[1];results[ch]={}
        for name in frozen['evaluated_names']:
            m=FusionModel.load(out/f'index509/{name}_{ch}.npz');digest=m.digest()
            f=m.extract(tr)['features'][mask];v=m.extract(va)['features'];lam=m.config['ridge']
            u,s,vh=np.linalg.svd(f,full_matrices=False)
            w=vh.T@((s/(s*s+lam))[:,None]*(u.T@target[mask]))
            delta=v@(w-m.Wout);diff=float(np.max(abs(delta)));scored_diff=float(np.max(abs(delta[mask])))
            residual=float(np.linalg.norm((f.T@f+lam*np.eye(f.shape[1]))@m.Wout-f.T@target[mask])/np.linalg.norm(f.T@target[mask]))
            # Evaluate protected interval separately from reset transients.
            assert scored_diff<1e-5,(name,ch,scored_diff)
            assert digest==m.digest()
            results[ch][name]={'validation_max_real_component_prediction_difference_vs_SVD':diff,'scored_validation_max_difference_vs_SVD':scored_diff,'relative_normal_equation_residual':residual}
    write_json(out/'independent_svd_validation_check.json',results)
    print('Independent SVD check passed; frozen weights unchanged.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('runs/phase3_feature_fusion'));a=p.parse_args();run(Path(__file__).resolve().parents[3],a.out.resolve())
