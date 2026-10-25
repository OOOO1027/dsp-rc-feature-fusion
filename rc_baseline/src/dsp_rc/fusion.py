"""Phase 3 feature fusion; original modules are intentionally unchanged."""
import json
import time
from pathlib import Path
import numpy as np
from .core import Reservoir, symbol_features, fit_ridge, array_hash


class LinearReservoir:
    """Identity reservoir with a certified Euclidean contraction.

    Scale A by its largest singular value, not its eigenvalue radius. Thus
    ||A||_2 = norm_bound < 1, ||(1-gamma)I+gamma A||_2 <= 1-gamma+gamma*norm_bound.
    norm_bound is NOT the actual spectral radius; both are reported.
    """
    def __init__(self, input_dim, nodes=80, leak=.7, norm_bound=.6, seed=200):
        if nodes < 1 or not 0 < leak <= 1 or not 0 <= norm_bound < 1:
            raise ValueError('Require nodes>0, 0<leak<=1 and 0<=norm_bound<1')
        self.input_dim, self.nodes, self.leak = input_dim, nodes, leak
        self.norm_bound = norm_bound
        rng = np.random.default_rng(seed)
        a = rng.uniform(-1, 1, (nodes, nodes))
        self.A = a * (norm_bound / np.linalg.norm(a, 2))
        self.Win = rng.uniform(-1, 1, (nodes, input_dim+1))
        self.verify_stability()

    def verify_stability(self):
        b = (1-self.leak)*np.eye(self.nodes)+self.leak*self.A
        norm = float(np.linalg.norm(b, 2))
        if not np.isfinite(norm) or norm >= 1:
            raise ValueError('Linear effective transition is not contractive')
        return {'A_spectral_radius':float(max(abs(np.linalg.eigvals(self.A)))),
                'A_operator_norm':float(np.linalg.norm(self.A,2)),
                'effective_spectral_radius':float(max(abs(np.linalg.eigvals(b)))),
                'effective_operator_norm':norm,
                'contraction_bound':1-self.leak+self.leak*self.norm_bound,
                'initial_state_residual_bound_128':(1-self.leak+self.leak*self.norm_bound)**128}

    def transform(self, x, initial_state=None):
        x = np.asarray(x)
        state = np.zeros(self.nodes) if initial_state is None else np.asarray(initial_state).copy()
        states = np.empty((len(x), self.nodes))
        drive = np.column_stack([x,np.ones(len(x))]) @ self.Win.T
        for t in range(len(x)):
            state = (1-self.leak)*state+self.leak*(self.A@state+drive[t])
            states[t] = state
        if not np.isfinite(states).all():
            raise FloatingPointError('Nonfinite linear states')
        return states, np.column_stack([states,x,np.ones(len(x))]), state.copy()


def candidates():
    base=dict(taps=115, linear_nodes=0, nonlinear_nodes=0, linear_norm=.6,
              linear_leak=.7, nonlinear_radius=.9, nonlinear_leak=.7,
              linear_seed=200, nonlinear_seed=100, passthrough=True, ridge=1e-5)
    configs=[]
    def add(name,family,**kw): configs.append(dict(base,**kw,name=name,family=family))
    for ridge,tag in [(1e-5,'low'),(.01,'reg')]:
        add('ridge_'+tag,'A',ridge=ridge)
        for n in [80,160]:
            add(f'tanh{n}_{tag}','B',nonlinear_nodes=n,ridge=ridge)
            add(f'linear{n}_{tag}','C',linear_nodes=n,ridge=ridge)
        for n in [40,80]:
            add(f'dual{n}_{tag}','D',linear_nodes=n,nonlinear_nodes=n,ridge=ridge)
    add('dual80_norm03','D',linear_nodes=80,nonlinear_nodes=80,linear_norm=.3)
    add('dual80_norm09','D',linear_nodes=80,nonlinear_nodes=80,linear_norm=.9)
    add('dual80_leak1','D',linear_nodes=80,nonlinear_nodes=80,linear_leak=1.)
    add('dual80_states_only','D',linear_nodes=80,nonlinear_nodes=80,passthrough=False)
    assert len(configs)==18
    return configs


class FusionModel:
    """fit(train z,train y,mask); transform(z) has no target or fitting path.

    Output features are [linear states, tanh states, optional input, one bias].
    Each transform resets both branch states. Input is a complete guarded split,
    not arbitrary chunks (centered windows require their own context).
    """
    def __init__(self, config): self.config=config.copy()
    def _build(self):
        c=self.config;d=2*c['taps'];self.linear=None;self.nonlinear=None
        if c['linear_nodes']:
            self.linear=LinearReservoir(d,c['linear_nodes'],c['linear_leak'],c['linear_norm'],c['linear_seed'])
        if c['nonlinear_nodes']:
            self.nonlinear=Reservoir(d,c['nonlinear_nodes'],c['nonlinear_leak'],c['nonlinear_radius'],c['nonlinear_seed'])
    def extract(self,z):
        x=symbol_features(z,self.config['taps'])/self.scale
        states={};last={};parts=[]
        for key,r in [('linear',self.linear),('nonlinear',self.nonlinear)]:
            if r is not None:
                states[key],_,last[key]=r.transform(x);parts.append(states[key])
        if self.config['passthrough']: parts.append(x)
        parts.append(np.ones((len(x),1)))
        return {'input':x,'states':states,'final_states':last,'features':np.column_stack(parts)}
    def fit(self,z,y,mask):
        t=time.perf_counter();x=symbol_features(z,self.config['taps'])
        self.scale=float(np.max(abs(x[mask])))
        if not self.scale>0: raise ValueError('Zero input scale')
        self._build();r=self.extract(z);f=r['features']
        self.Wout=fit_ridge(f[mask],np.column_stack([y.real,y.imag])[mask],self.config['ridge'])
        self.fit_seconds=time.perf_counter()-t
        self.stats={'feature_dim':f.shape[1],'trained_parameters':int(self.Wout.size),
                    'fixed_parameters':sum(int(v.A.size+v.Win.size) for v in [self.linear,self.nonlinear] if v is not None),
                    'input_scale':self.scale,'fit_seconds':self.fit_seconds,
                    'state_max_abs':{k:float(np.max(abs(v[mask]))) for k,v in r['states'].items()},
                    'linear_stability':self.linear.verify_stability() if self.linear else None}
        return self
    def transform(self,z):
        r=self.extract(z);o=r['features']@self.Wout
        r['prediction']=o[:,0]+1j*o[:,1]
        return r
    def predict(self,z): return self.transform(z)['prediction']
    def digest(self):
        arrays=[self.Wout,np.array([self.scale])]
        for r in [self.linear,self.nonlinear]:
            if r is not None: arrays.extend([r.A,r.Win])
        return array_hash(*arrays)
    def save(self,path):
        a={'config':json.dumps(self.config,sort_keys=True),'Wout':self.Wout,'scale':self.scale,
           'stats':json.dumps(self.stats)}
        for key,r in [('linear',self.linear),('nonlinear',self.nonlinear)]:
            if r is not None: a[key+'_A']=r.A;a[key+'_Win']=r.Win
        np.savez_compressed(path,**a)
    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as a:
            m=cls(json.loads(str(a['config'])));m._build();m.scale=float(a['scale']);m.Wout=a['Wout'];m.stats=json.loads(str(a['stats']))
            for key,r in [('linear',m.linear),('nonlinear',m.nonlinear)]:
                if r is not None: r.A=a[key+'_A'];r.Win=a[key+'_Win']
            if m.linear: m.linear.verify_stability()
        return m
