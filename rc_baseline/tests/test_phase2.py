import unittest
import numpy as np
from dsp_rc.phase2 import RCModel, candidate_list
from dsp_rc.core import decode_nltcp, ber_metrics


class Phase2Checks(unittest.TestCase):
    def test_forbidden_transitions_exhaustive(self):
        forbidden=[]
        for previous in range(6):
            for current in range(6):
                z=(np.array([previous,current])*2-5)*(1+1j)
                q,d=decode_nltcp(z)
                expected=abs(current-previous//2)
                self.assertEqual(d[1,0],expected)
                if expected>3:forbidden.append((previous,current))
        self.assertEqual(forbidden,[(0,4),(0,5),(1,4),(1,5),(2,5),(3,5)])

    def test_fixed_bit_bounds_exhaustive_unknown_mapping(self):
        z=np.array([-5-5j,5+5j]);labels=np.array([[0,0],[2,1]])
        result=ber_metrics(z,labels,np.array([False,True]))
        for a in range(4):
            for b in range(4):
                error=((a^2).bit_count()+(b^1).bit_count())/4
                self.assertLessEqual(result['ber_lower'],error)
                self.assertGreaterEqual(result['ber_upper'],error)

    def test_candidate_fit_isolated_and_frozen(self):
        rng=np.random.default_rng(17);train=rng.normal(size=400)+1j*rng.normal(size=400)
        target=.7*train+.1*np.roll(train,1);mask=np.arange(400)>128;mask[272:]=False
        c=candidate_list()[0];model=RCModel(c).fit(train,target,mask);before=model.digest()
        heldout=rng.normal(size=400)+1j*rng.normal(size=400)
        pred=model.predict(heldout);model.predict(-heldout)
        self.assertEqual(before,model.digest());np.testing.assert_array_equal(pred,model.predict(heldout))
        second=RCModel(c).fit(train,target,mask);self.assertEqual(second.digest(),before)

if __name__=='__main__':unittest.main()
