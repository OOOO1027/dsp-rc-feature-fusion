import unittest,tempfile
from pathlib import Path
import numpy as np
from experiment import LinearWaveLMS,Frontend,config
from dsp_rc.core import windows

class AblationTests(unittest.TestCase):
    def test_linear_update_independent_recursion(self):
        rng=np.random.default_rng(4);x=rng.normal(size=400);d=.6*x+.1*np.roll(x,1)
        m=LinearWaveLMS(5,.01).fit(x,d);features=windows(x,5)[:,::-1];w=np.zeros(5)
        for t in range(2,398):w=w+.01*(d[t]-features[t]@w)*features[t]
        np.testing.assert_allclose(m.W,w,atol=1e-15,rtol=1e-14)
        self.assertFalse(hasattr(m,'V'));self.assertEqual(len(m.errors),396)
    def test_no_quadratic_and_immutable_predict(self):
        rng=np.random.default_rng(7);x=rng.normal(size=800);m=LinearWaveLMS(7,.01).fit(x,.5*x)
        w=m.W.copy();np.testing.assert_array_equal(m.predict(2*x),2*m.predict(x))
        np.testing.assert_array_equal(m.predict(np.zeros(100)),np.zeros(100));np.testing.assert_array_equal(w,m.W)
    def test_scales_frozen_and_reload(self):
        rng=np.random.default_rng(9);x=rng.normal(scale=.2,size=12000);tx=.4*x
        for mode in ['on','off']:
            f=Frontend('rx1',mode,config()).fit(x,tx,dict(lag=0,polarity=1));digest=f.digest()
            val=rng.normal(scale=.4,size=4000);pred=f.preprocess(val,3000)[1]
            f.preprocess(val*2,3000);self.assertEqual(f.digest(),digest);self.assertEqual(pred.shape,(1000,))
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'front.npz';f.save(p);loaded=Frontend.load(p)
                np.testing.assert_array_equal(pred,loaded.preprocess(val,3000)[1]);self.assertEqual(loaded.digest(),digest)
    def test_segment_boundary_no_wrap(self):
        from dsp_rc.core import align_segment
        x=np.arange(100.)
        np.testing.assert_array_equal(align_segment(x,3,-1)[-3:],np.zeros(3))
        np.testing.assert_array_equal(align_segment(x,-3,1)[:3],np.zeros(3))
        np.testing.assert_array_equal(align_segment(x,3,-1)[:-3],-x[3:])
if __name__=='__main__':unittest.main()
