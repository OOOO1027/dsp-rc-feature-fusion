"""Read-only bundle integrity and optional frozen validation replay. Never fit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay-validation',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text())
    for rel,expected in manifest.items():
        assert sha(ROOT/rel)==expected, f'Changed or missing: {rel}'
    changes=json.loads((ROOT/'archive/portability_changes.json').read_text())
    imported=json.loads((ROOT/'archive/import_manifest.json').read_text())
    for row in imported:
        rel=row['path'];actual=sha(ROOT/rel)
        if rel in changes:
            assert row['sha256']==changes[rel]['original_sha256']
            assert actual==changes[rel]['delivered_sha256']
        else:assert actual==row['sha256'], f'Historical content changed: {rel}'
    sys.path.insert(0,str(ROOT/'rc_baseline/src'))
    from dsp_rc.portable import load_audit_index
    audit=load_audit_index(ROOT);count=0
    for folder in ['NN数据20250320','Data']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file():
                assert sha(path)==audit[str(path.resolve())]['sha256'],str(path)
                count+=1
    result={'bundle_files_verified':len(manifest),'imported_files_verified':len(imported),'audited_raw_files_verified':count,'real_data_training':False,'test_evaluation':False}
    if args.replay_validation:
        import numpy as np
        from dsp_rc.experiment import ChannelPipeline
        from dsp_rc.fusion import FusionModel
        # Use the prefix-only loader: no old test data parsed for inference.
        sys.path.insert(0,str(ROOT/'rc_baseline/volterra_ablation'))
        from experiment import load_prefix
        def forbidden(*args,**kwargs):raise RuntimeError('Training forbidden in read-only replay')
        ChannelPipeline.fit=forbidden;FusionModel.fit=forbidden
        data,_,_,_=load_prefix()
        config=json.loads((ROOT/'rc_baseline/runs/index509_fixed_v1/config.json').read_text())
        with np.load(ROOT/'rc_baseline/runs/phase3_feature_fusion/validation_predictions.npz') as saved:
            errors={}
            for ch in ['rx1','rx2']:
                front=ChannelPipeline.load(ROOT/f'rc_baseline/runs/index509_fixed_v1/model_{ch}.npz',ch,config)
                z=front.preprocess(data[ch][40000:80000],10000)[1]
                model=FusionModel.load(ROOT/f'rc_baseline/runs/phase3_feature_fusion/index509/dual40_reg_{ch}.npz')
                pred=model.predict(z);ref=saved[f'dual40_reg_{ch}']
                np.testing.assert_allclose(pred,ref,rtol=1e-10,atol=1e-10)
                errors[ch]=float(np.max(np.abs(pred-ref)))
            result['validation_prediction_max_error']=errors
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
