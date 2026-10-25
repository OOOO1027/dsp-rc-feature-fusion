"""Prepare/compare shared training-only MATLAB fixtures; never optimize."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.io import savemat,loadmat
from .experiment import ChannelPipeline,write_json
from .core import Reservoir,symbol_features,fit_ridge,rms,cap_demodulate,decode_nltcp,SymbolLMS,VolterraLMS


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','compare']);a=p.parse_args()
    base=Path(__file__).resolve().parents[2];out=base/'runs/phase2_509';old=base/'runs/index509_fixed_v1'
    c=json.loads((old/'config.json').read_text());root=base.parent/'NN数据20250320/NN数据20250320'
    if a.mode=='prepare':
        pipeline=ChannelPipeline.load(old/'model_rx2.npz','rx2',c)
        raw=np.loadtxt(root/'rxdata/rx2_PBS_NLTCP509.txt')[:40000];wave,z=pipeline.preprocess(raw,0)
        y=np.loadtxt(root/'v1.txt').ravel()+1j*np.loadtxt(root/'v2.txt').ravel()
        z=z[:512];x=symbol_features(z,25)/pipeline.input_scale
        savemat(out/'parity_input.mat',dict(x=x,z=z[:,None],A=pipeline.reservoir.A,Win=pipeline.reservoir.Win,
            gamma=.7,ridge=1e-5,washout=13,input_scale=pipeline.input_scale,
            target=np.column_stack([y[:512].real,y[:512].imag]),symbols=y[:512,None],wave=wave[:2048,None],
            txwave=np.loadtxt(root/'txdata/RF0.105_data36QAM.txt')[:2048,None]))
    else:
        d=loadmat(out/'parity_input.mat',squeeze_me=True);m=loadmat(out/'parity_matlab.mat',squeeze_me=True)
        r=Reservoir(50);r.A=d['A'];r.Win=d['Win'];states,features,_=r.transform(d['x'])
        w=fit_ridge(features[13:],d['target'][13:],1e-5);pred=features@w
        cap=cap_demodulate(d['wave']);cap/=rms(cap)
        q,decoded=decode_nltcp(d['symbols'])
        z=d['z']/rms(d['z']);y=d['symbols']/rms(d['symbols']);mask=np.zeros(len(z),bool);mask[4:252]=True
        lms=SymbolLMS(9,.001).fit(z,y,mask);lo=lms.predict(z);lo[:4]=z[:4];lo[-4:]=z[-4:];lo/=rms(lo)
        wave=d['wave']/rms(d['wave']);tx=d['txwave']/rms(d['txwave'])
        vol=VolterraLMS(9,.001,5,.00001).fit(wave[:512],tx[:512]);vo=vol.predict(wave);vo[:4]=wave[:4];vo[-4:]=wave[-4:];vo/=rms(vo)
        pairs={'states':(states,m['states']),'features':(features,m['features']),
            'readout':(w,m['Wout'].T),'pred':(pred,m['pred']),'window':(d['x'],m['x_window']),
            'q':(q,np.column_stack([m['q_i'],m['q_q']])),
            'decoded':(decoded,np.column_stack([m['d_i'],m['d_q']])),
            'lms_weights':(lms.W,m['lms_W']),'lms_output':(lo,m['lms_output']),
            'volterra_weights':(vol.W,m['vol_W']),'volterra_quadratic':(vol.V,m['vol_V']),
            'volterra_output':(vo,m['vol_output'])}
        report={}
        if m['cap'].size:pairs['cap']=(cap,m['cap'])
        else:report['cap']={'passed':None,'reason':str(m['cap_status'])}
        for name,(x,y) in pairs.items():
            error=float(np.max(abs(x-y)));np.testing.assert_allclose(x,y,rtol=2e-7,atol=2e-8)
            report[name]={'max_abs_difference':error,'passed':True}
        if np.size(m['bit_errors']):
            assert int(m['bit_errors'])==3 and abs(float(m['bit_rate'])-3/18)<1e-14
            report['biterr_auto_width']={'matlab_errors':int(m['bit_errors']),'matlab_ber':float(m['bit_rate']),'width':3,'passed':True}
        else:report['biterr_auto_width']={'passed':None,'reason':str(m['biterr_status'])}
        r0=Reservoir(50,seed=100)
        report['same_seed_different_rng']={'A_max_abs_difference':float(max(abs(r0.A-m['rng_A']).ravel())),
                                         'Win_max_abs_difference':float(max(abs(r0.Win-m['rng_Win']).ravel()))}
        write_json(out/'matlab_parity_results.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
