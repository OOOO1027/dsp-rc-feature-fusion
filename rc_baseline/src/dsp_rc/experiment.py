from dsp_rc.portable import load_audit_index
"""Reproducible train/validation/frozen-test experiment runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from .core import (rms, align_parameters, align_segment, cap_demodulate,
                   symbol_features, VolterraLMS, SymbolLMS, Reservoir,
                   fit_ridge, array_hash, decode_nltcp, ber_metrics)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024*1024): h.update(block)
    return h.hexdigest()


def mask_for(n, guard):
    if n <= 2*guard: raise ValueError('Split too short')
    mask = np.zeros(n, dtype=bool); mask[guard:n-guard] = True
    return mask


def check_config(c):
    if c['setting'] != 'S2S': raise ValueError('This implementation is S2S only')
    splits = list(c['splits'].values())
    assert all(a >= 0 and b > a for a,b in splits)
    assert all(splits[i][1] <= splits[i+1][0] for i in range(len(splits)-1))
    # Conservative sum of alignment, waveform filter, CAP filter, symbol LMS/RC,
    # RC washout and the previous NLTCP decision; covers nested centered contexts.
    wave_half = max(max(v[0], v[2])//2 for v in c['frontend'].values())
    context = int(np.ceil((c['alignment_max_lag']+wave_half)/c['sps']))
    context += c['cap']['span']//2
    context += max(c['rc']['window_symbols']//2, c['symbol_lms']['taps']//2)
    context += c['rc']['washout']+1
    assert c['guard_symbols'] >= context, (c['guard_symbols'], context)
    return context


def load_assets(root, c):
    nroot = root/'NN数据20250320'/'NN数据20250320'
    paths = {
        'rx1': nroot/'rxdata'/f"rx1_PBS_NLTCP{c['index']}.txt",
        'rx2': nroot/'rxdata'/f"rx2_PBS_NLTCP{c['index']}.txt",
        'tx': nroot/'txdata'/'RF0.105_data36QAM.txt',
        'i': nroot/'v1.txt', 'q': nroot/'v2.txt',
        'pam_i': nroot/'decimal_PAM4_1.txt', 'pam_q': nroot/'decimal_PAM4_2.txt',
    }
    # Check against the existing audit instead of guessing alternative files.
    import csv
    audit = load_audit_index(root)
    manifest = {}
    for name, p in paths.items():
        h = file_hash(p)
        if h != audit[str(p)]['sha256']: raise RuntimeError(f'Original asset changed: {p}')
        manifest[name] = {'path': str(p), 'sha256': h, 'bytes': p.stat().st_size}
    a = {k: np.loadtxt(p).ravel() for k,p in paths.items()}
    assert len(a['tx']) == len(a['rx1']) == len(a['rx2']) == 204800
    assert all(len(a[k]) == 51200 for k in ['i', 'q', 'pam_i', 'pam_q'])
    assert all(np.isfinite(v).all() for v in a.values())
    assert all(np.isin(a[k], np.arange(4)).all() for k in ['pam_i','pam_q'])
    return a, manifest


def scores(z, y, labels, mask):
    result = ber_metrics(z, labels, mask)
    result['symbol_mse'] = float(np.mean(abs(z[mask]-y[mask])**2))
    return result


class ChannelPipeline:
    """Frozen preprocessing + independent single-reservoir + LMS comparator.

    fit consumes one training split only. transform exposes raw normalized
    CAP symbols, RC inputs, states and readout features for future fusion.
    """
    def __init__(self, channel, config):
        self.channel = channel; self.c = config

    def preprocess(self, raw, start_symbol):
        c = self.c
        aligned = align_segment(raw, **{k:self.alignment[k] for k in ['lag','polarity']})/self.rx_scale
        wave = self.frontend.predict(aligned)/self.wave_scale
        symbols = cap_demodulate(wave, start_symbol*c['sps'], c['sps'], **self.c['cap'])/self.cap_scale
        return wave, symbols

    def fit(self, rx_train, tx_train, y_train):
        c = self.c; guard = c['guard_symbols']; mask = mask_for(len(y_train),guard)
        count = min(c['alignment_train_samples'],len(rx_train))
        self.alignment = align_parameters(rx_train[:count],tx_train[:count],c['alignment_max_lag'])
        rx = align_segment(rx_train,self.alignment['lag'],self.alignment['polarity'])
        self.rx_scale = rms(rx[guard*c['sps']:-guard*c['sps']])
        self.tx_scale = rms(tx_train[guard*c['sps']:-guard*c['sps']])
        nt = min(c['frontend_train_samples'],len(rx))
        self.frontend = VolterraLMS(*c['frontend'][self.channel]).fit(rx[:nt]/self.rx_scale, tx_train[:nt]/self.tx_scale)
        wave = self.frontend.predict(rx/self.rx_scale)
        self.wave_scale = rms(wave[guard*c['sps']:-guard*c['sps']])
        z = cap_demodulate(wave/self.wave_scale,0,c['sps'],**c['cap'])
        self.cap_scale = rms(z[mask]); z /= self.cap_scale
        self.target_scale = rms(y_train[mask])
        x = symbol_features(z,c['rc']['window_symbols'])
        self.input_scale = float(np.max(np.abs(x[mask])))
        x /= self.input_scale
        r = c['rc']
        self.reservoir = Reservoir(x.shape[1],r['nodes'],r['leak'],r['spectral_radius'],c['seed'])
        states, features, _ = self.reservoir.transform(x)
        target = np.column_stack([y_train.real,y_train.imag])
        self.Wout = fit_ridge(features[mask],target[mask],r['ridge'])
        self.lms = SymbolLMS(c['symbol_lms']['taps'],c['symbol_lms'][self.channel+'_step']).fit(z,y_train/self.target_scale,mask)
        # A training-only complex gain makes pre-symbol-equalizer scatter readable.
        self.before_gain = np.vdot(z[mask],y_train[mask])/np.vdot(z[mask],z[mask])
        # Report lag diagnostic, do not use held-out labels for changes.
        correlations = {}
        for lag in range(-8,9):
            t=np.arange(guard,len(y_train)-guard)
            correlations[str(lag)] = float(abs(np.vdot(z[t+lag],y_train[t]))/np.sqrt(np.sum(abs(z[t+lag])**2)*np.sum(abs(y_train[t])**2)))
        self.symbol_alignment_diagnostic = correlations
        self.training_mse = float(np.mean((features[mask]@self.Wout-target[mask])**2))
        self.ridge_residual = float(np.linalg.norm((features[mask].T@features[mask]+r['ridge']*np.eye(features.shape[1]))@self.Wout-features[mask].T@target[mask])/np.linalg.norm(features[mask].T@target[mask]))
        return self

    def transform(self, raw, start_symbol):
        _, z = self.preprocess(raw,start_symbol)
        x = symbol_features(z,self.c['rc']['window_symbols'])/self.input_scale
        states, features, final_state = self.reservoir.transform(x)
        o = features@self.Wout
        return {'before':z*self.before_gain, 'lms':self.lms.predict(z)*self.target_scale,
                'rc':o[:,0]+1j*o[:,1], 'input':x, 'states':states,
                'features':features, 'final_state':final_state, 'cap_symbols':z}

    def frozen_hash(self):
        return array_hash(self.frontend.W,self.frontend.V,self.lms.W,self.reservoir.A,self.reservoir.Win,self.Wout,
                          np.array([self.rx_scale,self.tx_scale,self.wave_scale,self.cap_scale,self.target_scale,self.input_scale,self.before_gain,self.alignment['lag'],self.alignment['polarity']]))

    def save(self,path):
        np.savez_compressed(path,frontend_W=self.frontend.W,frontend_V=self.frontend.V,
            lms_W=self.lms.W,A=self.reservoir.A,Win=self.reservoir.Win,Wout=self.Wout,
            scales=np.array([self.rx_scale,self.tx_scale,self.wave_scale,self.cap_scale,self.target_scale,self.input_scale]),
            before_gain=self.before_gain,lag=self.alignment['lag'],polarity=self.alignment['polarity'],
            frontend_errors=self.frontend.errors,lms_errors=self.lms.errors)

    @classmethod
    def load(cls,path,channel,config):
        obj=cls(channel,config)
        with np.load(path) as m:
            obj.frontend=VolterraLMS(*config['frontend'][channel]);obj.frontend.W=m['frontend_W'];obj.frontend.V=m['frontend_V']
            obj.lms=SymbolLMS(config['symbol_lms']['taps'],config['symbol_lms'][channel+'_step']);obj.lms.W=m['lms_W']
            r=config['rc'];obj.reservoir=Reservoir(2*r['window_symbols'],r['nodes'],r['leak'],r['spectral_radius'],config['seed'])
            obj.reservoir.A=m['A'];obj.reservoir.Win=m['Win'];obj.Wout=m['Wout']
            obj.rx_scale,obj.tx_scale,obj.wave_scale,obj.cap_scale,obj.target_scale,obj.input_scale=map(float,m['scales'])
            obj.before_gain=complex(m['before_gain']);obj.alignment={'lag':int(m['lag']),'polarity':int(m['polarity'])}
        return obj


def make_plots(out, pred, y, mask, metrics, pipelines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    indices=np.flatnonzero(mask)[::max(1,int(mask.sum())//4000)]
    for channel in ['rx1','rx2','xy']:
        fig, axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,method,title in zip(axes,['before','lms','rc'],['Before symbol equalization','Symbol LMS','Single RC (S2S)']):
            z=pred[channel][method][indices]
            ax.scatter(z.real,z.imag,s=2,alpha=.23,rasterized=True)
            target=np.unique(y[mask]);ax.scatter(target.real,target.imag,s=18,marker='+',c='black',label='Ideal')
            ax.set(xlim=(-8,8),ylim=(-8,8),xlabel='I',ylabel='Q',title=title);ax.set_aspect('equal');ax.grid(alpha=.2)
        fig.suptitle(f'Index 509 | {channel.upper()} | held-out test | fixed training scales')
        fig.savefig(out/f'constellation_{channel}.png',dpi=160);plt.close(fig)
    fig, ax=plt.subplots(figsize=(8,4),layout='constrained')
    for ch in ['rx1','rx2','xy']:
        vals=[metrics[ch][m]['matlab_compatible_ber'] for m in ['before','lms','rc']]
        ax.plot(['Before','LMS','RC'],vals,'o-',label=ch.upper())
    ax.set(yscale='log',ylabel='MATLAB-compatible BER (auto bit width)',title='Index 509 | held-out test | equal scoring mask')
    ax.grid(alpha=.25);ax.legend();fig.savefig(out/'ber_comparison.png',dpi=160);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ch,p in pipelines.items():
        for ax,e in zip(axes,[p.frontend.errors,p.lms.errors]):
            loss=np.abs(e)**2;avg=np.convolve(loss,np.ones(200)/200,'valid')
            ax.semilogy(np.arange(len(avg))+200,avg,label=ch)
    for ax,title in zip(axes,['Waveform LMS/Volterra training','Symbol LMS training']):
        ax.set(xlabel='Training update',ylabel='MSE (200-update mean)',title=title);ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/'training_errors.png',dpi=160);plt.close(fig)


def run(root, config_path, out, smoke=False):
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    log=(out/'run.log').open('w',buffering=1)
    def emit(msg):
        text=f'[{time.perf_counter()-start:.3f}s] {msg}';print(text,flush=True);log.write(text+'\n')
    c=json.loads(config_path.read_text())
    if smoke:
        c['splits']={'train':[0,3000],'validation':[3000,4000],'development_check':[4000,5000]}
    required_guard=check_config(c);write_json(out/'config.json',c)
    write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                                     'utc_start':datetime.now(timezone.utc).isoformat(),'smoke':smoke})
    emit(f"Setting={c['setting']}; configuration fixed; guard={c['guard_symbols']} >= {required_guard}")
    assets, manifest=load_assets(root,c);write_json(out/'input_manifest.json',manifest)
    y=assets['i']+1j*assets['q'];labels=np.column_stack([assets['pam_i'],assets['pam_q']]).astype(int)
    train_start,train_end=c['splits']['train']; assert train_start==0
    # Small reference sanity check uses training symbols only, not held-out labels.
    _,decoded=decode_nltcp(y[:train_end]);check=np.arange(1,train_end)
    assert np.array_equal(decoded[check],labels[check]), 'Reference mapping fails inside training region'
    pipelines={};times={}
    for ch in ['rx1','rx2']:
        t=time.perf_counter()
        p=ChannelPipeline(ch,c).fit(assets[ch][:train_end*c['sps']],assets['tx'][:train_end*c['sps']],y[:train_end])
        pipelines[ch]=p;p.save(out/f'model_{ch}.npz');times[ch+'_fit_seconds']=time.perf_counter()-t
        emit(f'{ch} fitted; alignment={p.alignment}; RC real-component train MSE={p.training_mse:.6g}; ridge residual={p.ridge_residual:.2g}')
        emit(f'{ch} train symbol-correlation best lag={max(p.symbol_alignment_diagnostic,key=p.symbol_alignment_diagnostic.get)}')
    hashes={ch:p.frozen_hash() for ch,p in pipelines.items()}
    frozen={'config_sha256':file_hash(out/'config.json'),'model_hashes':hashes,
            'xy_weights':[0.5,0.5],'parameters_selected_on':'none; source defaults',
            'test_prediction_started':False,'utc':datetime.now(timezone.utc).isoformat()}
    # This artifact is written BEFORE any held-out prediction or evaluation.
    write_json(out/'frozen_before_evaluation.json',frozen)
    all_metrics={};prediction_artifacts={};split_summary={}
    for split,(a,b) in c['splits'].items():
        mask=mask_for(b-a,c['guard_symbols']); yy=y[a:b];ll=labels[a:b]
        pred={};all_metrics[split]={};t=time.perf_counter()
        for ch,p in pipelines.items():
            z=p.transform(assets[ch][a*c['sps']:b*c['sps']],a)
            # Verify serialized inference before scoring; uses only input waveforms.
            loaded=ChannelPipeline.load(out/f'model_{ch}.npz',ch,c)
            assert loaded.frozen_hash()==hashes[ch]
            if split=='validation':
                replay=loaded.transform(assets[ch][a*c['sps']:b*c['sps']],a)
                np.testing.assert_array_equal(z['rc'],replay['rc'])
            pred[ch]={k:z[k] for k in ['before','lms','rc']}
            all_metrics[split][ch]={method:scores(v,yy,ll,mask) for method,v in pred[ch].items()}
            for method,v in pred[ch].items():prediction_artifacts[f'{split}_{ch}_{method}']=v
        pred['xy']={method:(pred['rx1'][method]+pred['rx2'][method])/2 for method in ['before','lms','rc']}
        all_metrics[split]['xy']={method:scores(v,yy,ll,mask) for method,v in pred['xy'].items()}
        for method,v in pred['xy'].items():prediction_artifacts[f'{split}_xy_{method}']=v
        prediction_artifacts[f'{split}_mask']=mask
        prediction_artifacts[f'{split}_symbol_ids_0based']=np.arange(a,b)
        split_summary[split]={'range_0based_half_open':[a,b], 'score_range_1based_inclusive':[a+c['guard_symbols']+1,b-c['guard_symbols']],
                             'scored_symbols':int(mask.sum()),'reference_bits':int(mask.sum()*4)}
        times[split+'_predict_score_seconds']=time.perf_counter()-t
        emit(f'{split}: '+ '; '.join(f"{ch} RC BER_auto={all_metrics[split][ch]['rc']['matlab_compatible_ber']:.6g}, fixed-2bit bounds=[{all_metrics[split][ch]['rc']['ber_lower']:.6g},{all_metrics[split][ch]['rc']['ber_upper']:.6g}]" for ch in ['rx1','rx2','xy']))
        if split=='test':make_plots(out,pred,yy,mask,all_metrics[split],pipelines)
        assert all(p.frozen_hash()==hashes[ch] for ch,p in pipelines.items()), 'Weights changed during inference'
    np.savez_compressed(out/'predictions.npz',**prediction_artifacts)
    times['total_seconds']=time.perf_counter()-start
    write_json(out/'results.json',{'metrics':all_metrics,'splits':split_summary,'timings':times,
               'alignment':{ch:p.alignment for ch,p in pipelines.items()},
               'symbol_alignment_training':{ch:p.symbol_alignment_diagnostic for ch,p in pipelines.items()},
               'model_unchanged_after_inference':True,'serialized_validation_prediction_exact':True,
               'rc_training_mse_per_real_component':{ch:p.training_mse for ch,p in pipelines.items()},
               'ridge_relative_residual':{ch:p.ridge_residual for ch,p in pipelines.items()}})
    emit('Complete. Frozen model hashes unchanged; no test-driven parameter changes.');log.close()


def main():
    parser=argparse.ArgumentParser();base=Path(__file__).resolve().parents[2]
    parser.add_argument('--root',type=Path,default=base.parent)
    parser.add_argument('--config',type=Path,default=base/'configs/index509.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.root.resolve(),args.config.resolve(),args.out.resolve(),args.smoke)

if __name__=='__main__':main()
