import json
from pathlib import Path
import unittest
import numpy as np
from dsp_rc.core import *
from dsp_rc.experiment import check_config, mask_for


class ScientificChecks(unittest.TestCase):
    def test_reservoir_equation_against_scalar_reference(self):
        r=Reservoir(3,nodes=4,seed=9)
        x=np.arange(18).reshape(6,3)/20
        before=array_hash(r.A,r.Win)
        states,features,last=r.transform(x)
        expected=[];s=np.zeros(4)
        for row in x:
            inp=np.r_[row,1.]
            activation=np.array([sum(r.A[i,j]*s[j] for j in range(4))+sum(r.Win[i,j]*inp[j] for j in range(4)) for i in range(4)])
            s=.3*s+.7*np.tanh(activation);expected.append(s.copy())
        np.testing.assert_allclose(states,expected,atol=1e-14)
        np.testing.assert_array_equal(features[:,4:7],x)
        np.testing.assert_array_equal(features[:,-1],1)
        self.assertEqual(before,array_hash(r.A,r.Win))
        self.assertAlmostEqual(max(abs(np.linalg.eigvals(r.A))),.9)
        np.testing.assert_array_equal(last,states[-1])

    def test_ridge_closed_form_and_no_prediction_refit(self):
        rng=np.random.default_rng(4);f=rng.normal(size=(120,10));y=rng.normal(size=(120,2));lam=.01
        w=fit_ridge(f[:80],y[:80],lam)
        expected=y[:80].T@f[:80]@np.linalg.inv(f[:80].T@f[:80]+lam*np.eye(10))
        np.testing.assert_allclose(w,expected.T,atol=1e-12)
        h=array_hash(w);pred=f[80:]@w
        y[80:]=1e9
        np.testing.assert_array_equal(w,fit_ridge(f[:80],y[:80],lam))
        self.assertEqual(h,array_hash(w));self.assertTrue(np.isfinite(pred).all())

    def test_complex_lms_matlab_index_and_conjugation(self):
        x=np.array([1+1j,2-1j,.5j,-1+2j,1-2j,2+.3j]);d=x*.5
        mask=np.array([False,True,True,True,True,False])
        l=SymbolLMS(3,.01).fit(x,d,mask);w=np.zeros(3,dtype=complex)
        # MATLAB n=compensation: input x(n:-1:n-taps+1), target at center.
        for n in range(3,len(x)+1):
            u=x[n-3:n][::-1];e=d[n-2]-w@u;w=w+.01*e*np.conj(u)
        np.testing.assert_allclose(l.W,w,atol=1e-15)

    def test_volterra_triu_basis(self):
        x=np.linspace(-1,1,30);d=np.sin(x)
        v=VolterraLMS(5,.02,3,.001).fit(x,d)
        w=np.zeros(5);q=np.eye(3)*1e-7
        for t in range(2,28):
            u=x[t-2:t+3][::-1];z=x[t-1:t+2][::-1]
            quad=np.triu(np.outer(z,z));e=d[t]-w@u-np.sum(q*quad)
            w+=.02*e*u;q+=.001*e*quad
        np.testing.assert_allclose(v.W,w);np.testing.assert_allclose(v.V,q)
        for t in range(2,28):
            u=x[t-2:t+3][::-1];z=x[t-1:t+2][::-1]
            self.assertAlmostEqual(v.predict(x)[t],w@u+np.sum(q*np.triu(np.outer(z,z))))

    def test_phase_and_root_raised_cosine(self):
        h=rrc();np.testing.assert_allclose(h,h[::-1]);self.assertAlmostEqual(np.sum(h*h),1)
        self.assertEqual(len(h),41)
        x=np.zeros(800);x[400]=1
        a=cap_demodulate(x);manual=np.convolve(x*np.exp(-1j*.5625*2*np.pi*np.arange(800)/4),h,'same')[::4]
        np.testing.assert_allclose(a,manual,atol=1e-13)
        h=rrc(.25,10,4);self.assertTrue(np.isfinite(h).all())

    def test_alignment_polarity_uses_training_only(self):
        x=np.random.default_rng(3).normal(size=1000);rx=np.r_[np.zeros(7),-x[:-7]]
        p=align_parameters(rx,x,20);self.assertEqual(p['lag'],7);self.assertEqual(p['polarity'],-1)
        aligned=align_segment(rx,p['lag'],p['polarity']);np.testing.assert_array_equal(aligned[:-7],x[:-7])
        np.testing.assert_array_equal(aligned[-7:],0)

    def test_nltcp_exact_mapping_and_fixed_bit_count(self):
        q=np.array([0,1,2,3,4,5,4,3,2,1,0])
        z=(q*2-5)*(1+1j);labels=np.zeros((len(q),2),int);labels[1:]=abs(q[1:]-q[:-1]//2)[:,None]
        mask=np.arange(len(q))>0;stats=ber_metrics(z,labels,mask)
        self.assertEqual(stats['ber_exact'],0);self.assertEqual(stats['reference_bits'],40)
        z2=np.array([-5-5j,5+5j]);m=ber_metrics(z2,np.zeros((2,2),int),np.array([False,True]))
        self.assertEqual(m['invalid_pam4_decisions'],2);self.assertIsNone(m['ber_exact'])
        self.assertEqual(m['ber_lower'],0);self.assertEqual(m['ber_upper'],1)

    def test_symbol_window_order(self):
        z=np.arange(10)+1j*np.arange(10,20);f=symbol_features(z,3)
        np.testing.assert_array_equal(f[4],[3,13,4,14,5,15])

    def test_boundary_guard_and_disjoint_split_support(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/index509.json').read_text())
        context=check_config(c);self.assertLessEqual(context,c['guard_symbols'])
        for a,b in c['splits'].values():
            i=np.flatnonzero(mask_for(b-a,c['guard_symbols']))
            self.assertGreaterEqual(i.min()-context,0);self.assertLess(i.max()+context,b-a)
        c['guard_symbols']=1
        with self.assertRaises(AssertionError):check_config(c)

    def test_reservoir_chunk_continuity_and_reset(self):
        r=Reservoir(2,nodes=5);x=np.random.default_rng(1).normal(size=(50,2))
        all_s,_,_=r.transform(x);a,_,state=r.transform(x[:20]);b,_,_=r.transform(x[20:],state)
        np.testing.assert_allclose(np.r_[a,b],all_s,atol=1e-14)
        first,_,_=r.transform(x);np.testing.assert_array_equal(first,all_s)

if __name__=='__main__':unittest.main()
