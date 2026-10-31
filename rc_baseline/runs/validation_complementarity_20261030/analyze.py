"""Validation-only post-hoc complementarity; no model training or test I/O.
Run from anywhere with the existing rc_baseline/.venv Python.
"""
from pathlib import Path
import sys, json, csv, hashlib, itertools, os
import numpy as np
from scipy.stats import rankdata
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/dsp-complementarity-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT/'rc_baseline/src'))
from dsp_rc.core import decode_nltcp, ber_metrics
PROTO = json.loads((OUT/'protocol.json').read_text())
ACCESSES = []

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def arrhash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def read_members(relative, keys):
    assert 'test' not in relative.lower()
    with np.load(ROOT/relative, allow_pickle=False) as z:
        out = {}
        for k in keys:
            assert 'train' not in k and 'test' not in k
            out[k] = z[k]
            ACCESSES.append(dict(path=relative, member=k, shape=list(out[k].shape), sha256=arrhash(out[k])))
    return out

def read_validation_label(filename):
    p=ROOT/'NN数据20250320/NN数据20250320'/filename
    # Original labels are a single physical line. Stream tokens without reading
    # the held-out suffix; unbuffered reads stop at token 20000.
    values=[]; token=bytearray(); count=0
    with p.open('rb', buffering=0) as f:
        while count < 20000:
            ch=f.read(1)
            if not ch: raise ValueError('Reference shorter than validation end')
            if ch.isspace():
                if token:
                    if count >= 10000: values.append(int(float(token.decode())))
                    count+=1; token.clear()
            else: token.extend(ch)
    a=np.array(values)
    assert a.shape==(10000,) and np.isin(a,np.arange(4)).all()
    ACCESSES.append(dict(path=str(p.relative_to(ROOT)), rows_zero_based=[10000,20000],sha256=arrhash(a)))
    return a

def cells(ea, eb):
    return dict(both_correct=int((~ea & ~eb).sum()), nonlinear_rescues_linear=int((ea & ~eb).sum()),
                linear_rescues_nonlinear=int((~ea & eb).sum()), both_wrong=int((ea & eb).sum()))

def bitcost(d, lab):
    legal=d<=3
    pc=np.array([int(i).bit_count() for i in range(8)])
    low=np.where(legal,pc[np.bitwise_xor(d,lab)],0).sum(axis=1)
    upper=low+2*(~legal).sum(axis=1)
    return low,upper

def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])

def moving_boot(values, block):
    values=np.asarray(values); n=len(values)
    rng=np.random.default_rng(PROTO['bootstrap']['seed']+block)
    cs=np.r_[0,np.cumsum(values)]
    full=n//block;tail=n%block
    starts=rng.integers(0,n-block+1,size=(PROTO['bootstrap']['draws'],full))
    sums=(cs[starts+block]-cs[starts]).sum(axis=1)
    if tail:
        starts=rng.integers(0,n-tail+1,size=len(sums));sums+=cs[starts+tail]-cs[starts]
    ci=np.quantile(sums/n,[.025,.975])
    return dict(mean=float(values.mean()), ci95=ci.tolist(), block=block, draws=len(sums))

def runs(b):
    v=np.r_[False,b,False].astype(int); a=np.flatnonzero(np.diff(v)==1);e=np.flatnonzero(np.diff(v)==-1)
    lengths=e-a
    return dict(events=int(len(a)),max_length=int(max(lengths,default=0)),
                median_length=float(np.median(lengths)) if len(lengths) else 0.,
                singleton_events=int((lengths==1).sum()))

def auc(score, positive):
    positive=np.asarray(positive,bool);n1=int(positive.sum());n0=len(positive)-n1
    if not n1 or not n0:return None
    return float((rankdata(score)[positive].sum()-n1*(n1+1)/2)/(n1*n0))

def table_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Meaningful checks: decoder previous-state effect; invalid decisions not remapped;
# four-cell counting; source cost bounds agree with existing evaluation.
def synthetic_checks():
    q,d=decode_nltcp(np.array([-5+0j,5+0j]));assert d[1,0]==5
    # Selecting the second branch's already decoded output differs from switching raw symbols.
    a=np.array([-5-5j,-5-5j]);b=np.array([5+5j,5+5j])
    _,da=decode_nltcp(a);_,db=decode_nltcp(b);_,ds=decode_nltcp(np.array([a[0],b[1]]))
    assert (db[1]!=ds[1]).all()
    assert cells(np.array([0,1,0,1],bool),np.array([0,0,1,1],bool))==dict(both_correct=1,nonlinear_rescues_linear=1,linear_rescues_nonlinear=1,both_wrong=1)
    assert moving_boot(np.zeros(600),256)['ci95']==[0.,0.]
    return dict(passed=True,checks=['illegal preserved','decoder switching distinction','four cells','zero bootstrap'])


def main():
    assert not (OUT/'summary.json').exists(), 'Preserve existing completed analysis'
    checks=synthetic_checks()
    pairs=[PROTO['primary_pair']]+PROTO['fixed_sensitivity_pairs']; names=list(dict.fromkeys(sum(pairs,[])+PROTO['context_predictions']))
    keys=[f'{name}_{ch}' for name in names for ch in PROTO['channels']]+['mask','symbol_ids']
    pred=read_members('rc_baseline/runs/phase3_feature_fusion/validation_predictions.npz',keys)
    diag=read_members('rc_baseline/runs/phase4_residual_fusion/residual_diagnostics_arrays.npz',
                      ['ideal_validation','rx1_validation_input','rx2_validation_input','mask'])
    y=diag['ideal_validation'];mask=pred['mask'];ids=pred['symbol_ids'];n=int(mask.sum())
    assert y.shape==(10000,) and np.array_equal(ids,np.arange(10000,20000))
    expected=np.zeros(10000,bool);expected[128:-128]=True
    assert np.array_equal(mask,expected) and np.array_equal(mask,diag['mask'])
    labels=np.column_stack([read_validation_label('decimal_PAM4_1.txt'),read_validation_label('decimal_PAM4_2.txt')])
    tq,td=decode_nltcp(y);assert np.array_equal(td[mask],labels[mask])
    old=json.loads((ROOT/'rc_baseline/runs/phase3_feature_fusion/validation_results.json').read_text())
    old={r['config']['name']:r['validation'] for r in old}
    scores={};dec={};counts=[];summary=[];strata=[];blockrows=[];aucrows=[];events=[];sensitivity=[]
    for name in names:
        for ch in PROTO['channels']:
            p=pred[name+'_'+ch];assert p.shape==(10000,) and np.isfinite(p).all()
            q,d=decode_nltcp(p);dec[name,ch]=(q,d)
            m=ber_metrics(p,labels,mask);m['symbol_mse']=float(np.mean(abs(p[mask]-y[mask])**2))
            for k in ['ber_lower','ber_upper','matlab_compatible_ber','symbol_mse']:
                assert np.isclose(m[k],old[name][ch][k],rtol=1e-12,atol=1e-12),(name,ch,k)
            scores[name+'_'+ch]=m
    for pair_no,(la,na) in enumerate(pairs):
        pair=f'{la}__{na}'
        for ch in PROTO['channels']:
            pl=pred[la+'_'+ch];pn=pred[na+'_'+ch];ql,dl=dec[la,ch];qn,dn=dec[na,ch]
            el=(dl[mask]!=labels[mask]).any(axis=1);en=(dn[mask]!=labels[mask]).any(axis=1)
            c=cells(el,en);union=el|en;both=el&en;rescue=el&~en;harm=~el&en
            lowerl,upperl=bitcost(dl[mask],labels[mask]);lowern,uppern=bitcost(dn[mask],labels[mask])
            for name,lo,hi in [(la,lowerl,upperl),(na,lowern,uppern)]:
                assert int(lo.sum())==scores[name+'_'+ch]['bit_errors_lower']
                assert int(hi.sum())==scores[name+'_'+ch]['bit_errors_upper']
            qel=(ql[mask]!=tq[mask]).any(axis=1);qen=(qn[mask]!=tq[mask]).any(axis=1)
            row=dict(pair=pair,channel=ch,n=n,**c,linear_wrong=int(el.sum()),nonlinear_wrong=int(en.sum()),
                     rescue_fraction_of_linear_errors=float(rescue.sum()/max(el.sum(),1)),
                     error_jaccard=float(both.sum()/max(union.sum(),1)),
                     decoded_pair_disagreement=int((dl[mask]!=dn[mask]).any(axis=1).sum()),
                     same_pam6_pair_fraction=float(np.mean((ql[mask]==qn[mask]).all(axis=1))),
                     diagnostic_oracle_pair_error_rate=float(both.mean()),
                     linear_pair_error_rate=float(el.mean()), nonlinear_pair_error_rate=float(en.mean()),
                     error_indicator_correlation=corr(el,en),
                     residual_I_correlation=corr((pl-y)[mask].real,(pn-y)[mask].real),
                     residual_Q_correlation=corr((pl-y)[mask].imag,(pn-y)[mask].imag),
                     linear_mse=float(np.mean(abs(pl[mask]-y[mask])**2)),nonlinear_mse=float(np.mean(abs(pn[mask]-y[mask])**2)),
                     prediction_difference_rms=float(np.sqrt(np.mean(abs(pl[mask]-pn[mask])**2))),
                     linear_invalid=int((dl[mask]>3).sum()),nonlinear_invalid=int((dn[mask]>3).sum()),
                     linear_valid_wrong_rescued=int((rescue & (dl[mask]<=3).all(axis=1)).sum()),
                     linear_invalid_rescued=int((rescue & (dl[mask]>3).any(axis=1)).sum()))
            summary.append(row)
            for level,a,b in [('decoded_PAM4_pair',el,en),('predecoder_PAM6_pair',qel,qen),
                              ('decoded_PAM4_component',(dl[mask]!=labels[mask]).ravel(),(dn[mask]!=labels[mask]).ravel())]:
                counts.append(dict(pair=pair,channel=ch,level=level,n=len(a),**cells(a,b)))
            for size in [256,512]:
                for endpoint,v in [('decoded_pair_error_delta_N_minus_L',en.astype(float)-el),
                                   ('BER_lower_delta_N_minus_L',(lowern-lowerl)/4),('BER_upper_delta_N_minus_L',(uppern-upperl)/4),
                                   ('MSE_delta_N_minus_L',(abs(pn[mask]-y[mask])**2-abs(pl[mask]-y[mask])**2))]:
                    blockrows.append(dict(pair=pair,channel=ch,endpoint=endpoint,**moving_boot(v,size)))
            if pair_no:continue
            # All descriptive strata are fixed here; no best-bin rule will be deployed.
            rx=diag[ch+'_validation_input'] if ch!='xy' else (diag['rx1_validation_input']+diag['rx2_validation_input'])/2
            components=np.column_stack([pl.real,pl.imag]); dist=abs(components[:,:,None]-np.arange(-5,6,2))
            sd=np.sort(dist,axis=2);margin=(sd[:,:,1]-sd[:,:,0]).min(axis=1)
            margin=np.minimum(margin,np.r_[margin[0],margin[:-1]])[mask]
            features={'received_amplitude':abs(rx)[mask], 'received_first_difference':abs(rx-np.r_[rx[0],rx[:-1]])[mask], 'linear_margin_current_and_previous':margin}
            grouping={'time_quarter':np.minimum(np.arange(n)*4//n,3)}
            for feature,v in features.items():
                edges=np.quantile(v,[.2,.4,.6,.8]);groups=np.searchsorted(edges,v,side='right');grouping[feature]=groups
                only=rescue|harm
                aucrows.append(dict(channel=ch,feature=feature,positive='nonlinear_rescues_linear',population='exactly_one_branch_correct',n=int(only.sum()),auc_higher_feature_predicts_rescue=auc(v[only],rescue[only]),bin_edges=edges.tolist()))
            grouping['ideal_constellation']=(tq[mask,0]*6+tq[mask,1])
            for feature,groups in grouping.items():
                for group in sorted(np.unique(groups)):
                    take=groups==group;cc=cells(el[take],en[take]);size=int(take.sum())
                    strata.append(dict(channel=ch,feature=feature,group=int(group),n=size,**cc,
                        rescue_rate=float(rescue[take].mean()),harm_rate=float(harm[take].mean()),net_rescue=int(rescue[take].sum()-harm[take].sum())))
            for label,b in [('nonlinear_rescue',rescue),('linear_rescue',harm),('both_wrong',both)]:
                events.append(dict(channel=ch,event=label,**runs(b)))
            for start in range(0,n,256):
                stop=min(start+256,n);cc=cells(el[start:stop],en[start:stop])
                sensitivity.append(dict(channel=ch,start_id_zero_based=int(ids[mask][start]),n=stop-start,**cc))
    table_csv('four_cells.csv',counts);table_csv('pair_summary.csv',summary);table_csv('strata.csv',strata);table_csv('temporal_blocks_256.csv',sensitivity)
    dump('metrics.json',scores);dump('paired_block_bootstrap.json',blockrows);dump('observable_diagnostics.json',aucrows);dump('event_runs.json',events)
    dump('summary.json',summary);dump('input_access_manifest.json',ACCESSES)
    dump('verification.json',dict(**checks,scored_symbols=n,scored_global_ids_zero_based=[int(ids[mask][0]),int(ids[mask][-1])],
        reproduced_existing_metrics=len(scores),ideal_validation_decodes_exactly=True,test_prediction_files_read=False,test_metrics_computed=False,model_fits=0,
        initial_reader_incident='Failed preliminary line-based reference reader read the shared single-line PAM4_1 label container including its test suffix as text, but parsed no labels and computed no test metric. Replaced by unbuffered token streaming stopping at symbol 20000. See REPORT boundary disclosure.',
        oracle_is_not_deployable=True,seed_replication_available=False))
    # Charts are based only on primary pair and clearly identify diagnostic oracle.
    primary=summary[:3];fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    xx=np.arange(3)
    for i,(key,label) in enumerate([('both_wrong','Both wrong'),('nonlinear_rescues_linear','Nonlinear rescue'),('linear_rescues_nonlinear','Linear rescue')]):
        axs[0].bar(xx+(i-1)*.24,[r[key] for r in primary],width=.24,label=label)
    axs[0].set(xticks=xx,xticklabels=['RX1','RX2','XY'],ylabel='Decoded PAM4 pairs (count)',title='Validation only: where errors overlap');axs[0].legend(fontsize=8)
    for i,(key,label) in enumerate([('linear_pair_error_rate','Linear80'),('nonlinear_pair_error_rate','Tanh80'),('diagnostic_oracle_pair_error_rate','Truth-assisted oracle (not deployable)')]):
        axs[1].bar(xx+(i-1)*.24,[100*r[key] for r in primary],width=.24,label=label)
    axs[1].set(xticks=xx,xticklabels=['RX1','RX2','XY'],ylabel='Decoded pair error rate (%)',title='Oracle gap is potential, not achieved BER');axs[1].legend(fontsize=7)
    fig.savefig(OUT/'complementarity.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,7),layout='constrained')
    for ax,ch in zip(axs,PROTO['channels']):
        rr=[r for r in sensitivity if r['channel']==ch];x=[r['start_id_zero_based'] for r in rr]
        ax.plot(x,[r['nonlinear_rescues_linear'] for r in rr],label='Nonlinear rescue',marker='.',lw=1)
        ax.plot(x,[r['linear_rescues_nonlinear'] for r in rr],label='Linear rescue',marker='.',lw=1)
        ax.set(title=ch.upper(),ylabel='Count / block');ax.legend(fontsize=8)
    axs[-1].set_xlabel('Global symbol ID (zero based; validation only)');fig.savefig(OUT/'temporal_rescue.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for i,ch in enumerate(PROTO['channels']):
        for j,feature in enumerate(['received_amplitude','received_first_difference','linear_margin_current_and_previous']):
            rr=[r for r in strata if r['channel']==ch and r['feature']==feature];ax=axs[i,j]
            ax.plot(range(1,6),[r['rescue_rate']*100 for r in rr],'o-',label='Nonlinear rescue')
            ax.plot(range(1,6),[r['harm_rate']*100 for r in rr],'o-',label='Linear rescue')
            ax.set(title=ch.upper()+' / '+feature.replace('_',' '),xlabel='Validation quintile (descriptive)',ylabel='Event rate (%)');ax.title.set_fontsize(9)
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.savefig(OUT/'observable_strata.png',dpi=160);plt.close(fig)
    print(json.dumps(primary,indent=2))

if __name__=='__main__': main()
