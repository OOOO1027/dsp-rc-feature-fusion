"""Render an existing frozen result; never fits models or reads source labels."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    run=args.run;r=json.loads((run/'results.json').read_text())
    rows=[]
    for split,channels in r['metrics'].items():
        for channel,methods in channels.items():
            for method,m in methods.items():
                rows.append({'split':split,'channel':channel,'method':method,
                             **{k:m[k] for k in ['symbols','reference_bits','bit_errors_lower','bit_errors_upper','ber_lower','ber_upper','invalid_pam4_decisions','matlab_compatible_ber','symbol_mse']}})
    with (run/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if 'test' not in r['metrics']:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    colors=['#54788d','#b2733c','#5d8270']
    x=np.arange(3)
    for j,ch in enumerate(['rx1','rx2','xy']):
        items=[r['metrics']['test'][ch][m] for m in ['before','lms','rc']]
        low=np.array([m['ber_lower'] for m in items]);high=np.array([m['ber_upper'] for m in items]);mid=(low+high)/2
        ax.errorbar(x+(j-1)*.07,mid,yerr=np.vstack([mid-low,high-mid]),marker='o',capsize=5,label=ch.upper(),color=colors[j],linewidth=1.5)
    ax.set_xticks(x,['Before symbol equalization','Symbol LMS','Single RC (S2S)'])
    ax.set(yscale='log',ylabel='Bit error fraction: fixed 4 reference bits / complex symbol',title='Held-out test | bounds for undefined 2-bit outputs (not confidence intervals)')
    ax.grid(alpha=.2);ax.legend();fig.savefig(run/'ber_fixed_bit_bounds.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
