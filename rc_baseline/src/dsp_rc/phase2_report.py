"""Summarize frozen Phase-2 artifacts. No model fit/test prediction occurs here."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=Path(__file__).resolve().parents[2];out=base/'runs/phase2_509'
    candidates=json.loads((out/'candidate_results.json').read_text());confirm=json.loads((out/'confirmation_results.json').read_text())
    rows=[]
    for r in candidates:
        for split in ['train','validation']:
            m=r[split];rows.append({'split':split,'channel':r['channel'],'model':r['candidate']['name'],
            **{k:m[k] for k in ['matlab_compatible_ber','ber_lower','ber_upper','invalid_rate','invalid_pam4_decisions','symbol_mse']},
            'fit_seconds':r['fit_seconds'],'validation_inference_seconds':r['validation_inference_seconds']})
    for ch,models in confirm['metrics'].items():
        for name,m in models.items():
            rows.append({'split':'holdout_confirmation','channel':ch,'model':name,
            **{k:m[k] for k in ['matlab_compatible_ber','ber_lower','ber_upper','invalid_rate','invalid_pam4_decisions','symbol_mse']},
            'fit_seconds':'','validation_inference_seconds':''})
    with (out/'metrics_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    linear=json.loads((out/'linear_window_ablation.json').read_text())
    for ax,ch in zip(axes,['rx1','rx2']):
        rr=[next(r for r in candidates if r['channel']==ch and r['candidate']['name']==name)['validation']['ber_upper'] for name in ['original','window57','window115']]
        ll=[r['metrics']['ber_upper'] for r in linear if r['channel']==ch]
        ax.plot([25,57,115],rr,'o-',label='Single RC');ax.plot([25,57,115],ll,'s--',label='Linear ridge, no reservoir')
        ax.set(title=ch.upper()+' | validation only',xlabel='Window (complex symbols)',ylabel='Fixed 2-bit BER upper bound',yscale='log');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'window_diagnosis.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');x=np.arange(3)
    for i,ch in enumerate(['rx1','rx2','xy']):
        ms=[confirm['metrics'][ch][k] for k in ['old_rc','selected_rc','lms']]
        lo=np.array([m['ber_lower'] for m in ms]);hi=np.array([m['ber_upper'] for m in ms]);mid=(lo+hi)/2
        ax.errorbar(x+.07*(i-1),mid,yerr=np.array([mid-lo,hi-mid]),marker='o',capsize=4,label=ch.upper())
    ax.set_xticks(x,['Original RC (25)','Selected RC (115)','LMS (115)'])
    ax.set(yscale='log',ylabel='Fixed 2-bit BER bounds',title='Previously observed holdout confirmation | bounds are not confidence intervals')
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'confirmation_bounds.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
