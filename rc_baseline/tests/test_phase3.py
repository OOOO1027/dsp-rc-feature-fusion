import tempfile
import unittest
from pathlib import Path
import numpy as np
from dsp_rc.fusion import LinearReservoir, FusionModel, candidates
from dsp_rc.phase2 import RCModel, candidate_list

class FusionChecks(unittest.TestCase):
    def test_linear_closed_form_affine_and_contraction(self):
        r=LinearReservoir(3,7);x=np.random.default_rng(9).normal(size=(40,3))
        st,_,_=r.transform(x);b=(1-r.leak)*np.eye(r.nodes)+r.leak*r.A
        for t in [0,5,39]:
            expected=sum(np.linalg.matrix_power(b,t-j)@(r.leak*r.Win@np.r_[x[j],1]) for j in range(t+1))
            np.testing.assert_allclose(st[t],expected,atol=1e-13)
        a=r.transform(2*x)[0];zero=r.transform(np.zeros_like(x))[0]
        np.testing.assert_allclose(a-zero,2*(st-zero),atol=1e-13)
        shifted=r.transform(x,np.ones(r.nodes))[0]
        self.assertLess(np.linalg.norm(shifted[-1]-st[-1]),r.verify_stability()['contraction_bound']**40*np.sqrt(r.nodes))
    def test_unstable_parameters_and_matrix_rejected(self):
        for kw in [dict(norm_bound=1),dict(leak=0),dict(leak=1.2)]:
            with self.assertRaises(ValueError):LinearReservoir(3,**kw)
        r=LinearReservoir(3,4);r.A=2*np.eye(4)
        with self.assertRaises(ValueError):r.verify_stability()
    def test_exact_phase2_nonlinear_reproduction(self):
        rng=np.random.default_rng(12);z=rng.normal(size=400)+1j*rng.normal(size=400)
        y=.8*z+.2*np.roll(z,1);mask=np.zeros(400,bool);mask[128:-128]=True
        c=next(c for c in candidates() if c['name']=='tanh80_low')
        new=FusionModel(c).fit(z,y,mask)
        old=RCModel(next(c for c in candidate_list() if c['name']=='window115')).fit(z,y,mask)
        np.testing.assert_array_equal(new.predict(z),old.predict(z))
    def test_feature_dimensions_saved_inference_and_label_isolation(self):
        rng=np.random.default_rng(10);z=rng.normal(size=400)+1j*rng.normal(size=400);y=z*.7
        mask=np.zeros(400,bool);mask[128:-128]=True
        c=next(c for c in candidates() if c['name']=='dual40_low');m=FusionModel(c).fit(z,y,mask)
        self.assertEqual(m.extract(z)['features'].shape,(400,311))
        f=m.extract(z)['features'];np.testing.assert_array_equal(f[:,-231:-1],m.extract(z)['input'])
        y2=y.copy();y2[~mask]=999;other=FusionModel(c).fit(z,y2,mask)
        self.assertEqual(m.digest(),other.digest())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.npz';m.save(p);loaded=FusionModel.load(p)
            digest=m.digest();np.testing.assert_array_equal(m.predict(z),loaded.predict(z))
            m.predict(z[::-1]);self.assertEqual(digest,m.digest())
            np.testing.assert_array_equal(m.predict(z),loaded.predict(z))
    def test_branch_chunk_continuity(self):
        r=LinearReservoir(2,5);x=np.random.default_rng(10).normal(size=(90,2))
        whole=r.transform(x)[0];a,_,state=r.transform(x[:40]);b,_,_=r.transform(x[40:],state)
        np.testing.assert_allclose(whole,np.vstack([a,b]),atol=1e-14)

if __name__=='__main__':unittest.main()
