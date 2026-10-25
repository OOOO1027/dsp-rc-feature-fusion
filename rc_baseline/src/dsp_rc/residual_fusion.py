"""Phase 4: sequential residual prediction and explicitly transformed penalties.
All statistics/projections are fitted on the training mask only.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
from .core import fit_ridge, symbol_features, array_hash
from .fusion import FusionModel,candidates as phase3_candidates


def configurations():
    result=[]
    for ref in ['ridge_low','linear80_low','linear160_low','tanh80_low','dual40_reg','dual80_reg']:
        result.append(dict(name=ref,kind='reference',reference=ref,eta=None))
    for base in ['ridge_low','linear80_low']:
        for orth in [False,True]:
            for eta in [.25,.5,1.]:
                result.append(dict(name=f'res_{base}_{"orth" if orth else "raw"}_eta{eta:g}',kind='residual',base=base,orth=orth,eta=eta,penalty=.01))
    for orth in [False,True]:
        for mode in ['uniform','group','equivalent']:
            result.append(dict(name=f'joint_{"orth" if orth else "std"}_{mode}',kind='joint',orth=orth,penalty_mode=mode,eta=None))
    assert len(result)==24
    return result


def reference_config(name,seed=100):
    c=next(c.copy() for c in phase3_candidates() if c['name']==name)
    c['nonlinear_seed']=seed
    return c


def penalized_lstsq(f,y,penalty_factor):
    """Min ||FW-Y||²+||R W||² via augmented QR, never square conditioning."""
    if np.ndim(penalty_factor)==1: penalty_factor=np.diag(np.sqrt(penalty_factor))
    return lstsq(np.vstack([f,penalty_factor]),np.vstack([y,np.zeros((penalty_factor.shape[0],y.shape[1]))]),lapack_driver='gelsy')[0]


class FeatureBank:
    def __init__(self,seed=100):
        self.model=FusionModel(reference_config('dual80_reg',seed))
    def fit(self,z,mask):
        t=time.perf_counter();self.model.scale=float(np.max(abs(symbol_features(z,115)[mask])));self.model._build()
        self.fit_seconds=time.perf_counter()-t;return self
    def transform(self,z):
        r=self.model.extract(z);x=r['input'];l=r['states']['linear'];h=r['states']['nonlinear']
        c=np.column_stack([x,l,np.ones(len(x))])
        return c,h


def coordinate_transform(c,h,mask,orth=False):
    """Fnew=Fraw T. Ridge projection is approximate, not exact OLS orthogonality."""
    nc=c.shape[1];d=nc+h.shape[1];p=fit_ridge(c[mask],h[mask],.01) if orth else np.zeros((nc,h.shape[1]))
    t=np.eye(d);t[:nc,nc:]=-p;f=np.column_stack([c,h]);g=f@t
    mean=g[mask].mean(axis=0);scale=g[mask].std(axis=0)
    bias=nc-1;mean[bias]=0;scale[bias]=1;scale=np.maximum(scale,1e-12)
    s=np.diag(1/scale);s[bias,:]-=mean/scale
    return t@s,p,mean,scale


class ResidualFusion:
    def __init__(self,config,seed=100): self.config=config.copy();self.seed=seed
    def fit(self,z,y,mask,bank=None):
        t=time.perf_counter();self.bank=bank if bank is not None else FeatureBank(self.seed).fit(z,mask)
        c,h=self.bank.transform(z);target=np.column_stack([y.real,y.imag]);self.arrays={}
        cfg=self.config;self.base=None
        if cfg['kind']=='residual':
            self.base=FusionModel(reference_config(cfg['base'],self.seed)).fit(z,y,mask)
            pred=self.base.predict(z);res=target-np.column_stack([pred.real,pred.imag])
            self.arrays['projection']=fit_ridge(c[mask],h[mask],.01) if cfg['orth'] else np.zeros((c.shape[1],h.shape[1]))
            e=h-c@self.arrays['projection'];self.arrays['mean']=e[mask].mean(axis=0);self.arrays['std']=np.maximum(e[mask].std(axis=0),1e-12)
            v=np.column_stack([(e-self.arrays['mean'])/self.arrays['std'],np.ones(len(e))])
            self.arrays['W']=penalized_lstsq(v[mask],res[mask],np.full(v.shape[1],cfg['penalty']))
            self.arrays['base_W']=self.base.Wout.copy()
            outparams=self.base.Wout.size+self.arrays['W'].size;dimension=self.base.Wout.shape[0]+v.shape[1]
            learned_projection=int(self.arrays['projection'].size) if cfg['orth'] else 0
        else:
            transform,p,mean,std=coordinate_transform(c,h,mask,cfg['orth']);self.arrays.update(T=transform,projection=p,mean=mean,std=std)
            f=np.column_stack([c,h]);g=f@transform
            penalties=np.full(g.shape[1],.01)
            if cfg['penalty_mode']=='group':penalties[230:310]=1.;penalties[311:]=10.
            factor=np.sqrt(.01)*transform if cfg['penalty_mode']=='equivalent' else np.diag(np.sqrt(penalties))
            self.arrays['W']=penalized_lstsq(g[mask],target[mask],factor)
            self.arrays['penalty_factor']=factor
            outparams=self.arrays['W'].size;dimension=g.shape[1];learned_projection=int(p.size) if cfg['orth'] else 0
        self.fit_seconds=time.perf_counter()-t
        self.stats=dict(fit_seconds=self.fit_seconds,readout_parameters=int(outparams),readout_dimension=int(dimension),
                        projection_fitted_coefficients=learned_projection,normalization_fitted_scalars=int(2*self.arrays['mean'].size+1),
                        materialized_fixed_reservoir_coefficients=49760,bank_setup_seconds=self.bank.fit_seconds)
        return self
    def components(self,z):
        c,h=self.bank.transform(z);a=self.arrays
        if self.config['kind']=='residual':
            e=h-c@a['projection'];v=np.column_stack([(e-a['mean'])/a['std'],np.ones(len(e))]);correction=v@a['W']
            pred=self.base.predict(z)
            return pred,correction[:,0]+1j*correction[:,1]
        output=np.column_stack([c,h])@a['T']@a['W']
        return output[:,0]+1j*output[:,1],np.zeros(len(z),complex)
    def predict(self,z,eta=None):
        base,correction=self.components(z)
        return base+(self.config['eta'] if eta is None else eta)*correction if self.config['kind']=='residual' else base
    def digest(self):
        m=self.bank.model;arrays=[m.linear.A,m.linear.Win,m.nonlinear.A,m.nonlinear.Win,np.array([m.scale])]
        arrays += [self.arrays[k] for k in sorted(self.arrays)]
        return array_hash(*arrays)
    def save(self,path):
        m=self.bank.model
        np.savez_compressed(path,config=json.dumps(self.config),seed=self.seed,stats=json.dumps(self.stats),scale=m.scale,
            linear_A=m.linear.A,linear_Win=m.linear.Win,nonlinear_A=m.nonlinear.A,nonlinear_Win=m.nonlinear.Win,**self.arrays)
    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as a:
            m=cls(json.loads(str(a['config'])),int(a['seed']));m.stats=json.loads(str(a['stats']));m.bank=FeatureBank(m.seed)
            b=m.bank.model;b.scale=float(a['scale']);b._build();b.linear.A=a['linear_A'];b.linear.Win=a['linear_Win'];b.nonlinear.A=a['nonlinear_A'];b.nonlinear.Win=a['nonlinear_Win']
            keys={'config','seed','stats','scale','linear_A','linear_Win','nonlinear_A','nonlinear_Win'};m.arrays={k:a[k] for k in a.files if k not in keys}
            if m.config['kind']=='residual':
                m.base=FusionModel(reference_config(m.config['base'],m.seed));m.base.scale=b.scale;m.base._build();m.base.Wout=m.arrays['base_W']
                if m.base.linear is not None:m.base.linear.A=b.linear.A;m.base.linear.Win=b.linear.Win
            else:m.base=None
        return m
