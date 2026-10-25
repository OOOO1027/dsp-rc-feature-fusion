import unittest,tempfile
from pathlib import Path
import numpy as np
from dsp_rc.residual_fusion import ResidualFusion,configurations,penalized_lstsq,coordinate_transform
class ResidualChecks(unittest.TestCase):
 def test_coordinate_and_penalty_equivalence(self):
  rng=np.random.default_rng(4);c=np.column_stack([rng.normal(size=(300,6)),np.ones(300)]);h=rng.normal(size=(300,4));y=rng.normal(size=(300,2));mask=np.arange(300)<200;f=np.column_stack([c,h]);w=penalized_lstsq(f[mask],y[mask],np.full(11,.01))
  for orth in [False,True]:
   t,p,_,_=coordinate_transform(c,h,mask,orth);v=penalized_lstsq((f@t)[mask],y[mask],np.sqrt(.01)*t);np.testing.assert_allclose(f@w,f@t@v,atol=1e-12)
   if orth:np.testing.assert_allclose((c[mask].T@c[mask]+.01*np.eye(7))@p,c[mask].T@h[mask],atol=1e-10)
 def test_eta_zero_and_save_load(self):
  rng=np.random.default_rng(4);z=rng.normal(size=500)+1j*rng.normal(size=500);y=.7*z+.1*np.roll(z,1);mask=np.zeros(500,bool);mask[128:-128]=True
  cfg=next(c for c in configurations() if c['kind']=='residual' and c['orth'] and c['base']=='linear80_low');m=ResidualFusion(cfg).fit(z,y,mask);np.testing.assert_array_equal(m.predict(z,eta=0),m.base.predict(z));before=m.digest()
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.npz';m.save(path);loaded=ResidualFusion.load(path);np.testing.assert_array_equal(m.predict(z),loaded.predict(z));self.assertEqual(before,loaded.digest())
  m.predict(z[::-1]);self.assertEqual(before,m.digest())
 def test_targets_outside_training_mask_do_not_fit(self):
  rng=np.random.default_rng(7);z=rng.normal(size=500)+1j*rng.normal(size=500);y=.6*z;mask=np.zeros(500,bool);mask[128:-128]=True;y2=y.copy();y2[~mask]=9999
  for cfg in [configurations()[6],configurations()[-2]]:
   a=ResidualFusion(cfg).fit(z,y,mask);b=ResidualFusion(cfg).fit(z,y2,mask);self.assertEqual(a.digest(),b.digest())
 def test_standardization_training_only(self):
  rng=np.random.default_rng(4);c=np.column_stack([rng.normal(size=(200,3)),np.ones(200)]);h=rng.normal(size=(200,4));mask=np.arange(200)<100;a=coordinate_transform(c,h,mask,True);c[100:,:-1]*=1000;h[100:]*=1000;b=coordinate_transform(c,h,mask,True)
  for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
 def test_block_pairing_and_bit_denominator(self):
  from dsp_rc.phase4 import per_symbol,paired_blocks
  from dsp_rc.phase2 import enriched_scores
  z=np.tile(np.array([-5-5j,5+5j,-1+1j,3-3j]),30);labels=np.tile(np.array([[0,1],[2,3],[1,0],[3,2]]),(30,1));mask=np.ones(len(z),bool);mask[:3]=False;mask[-3:]=False
  losses=per_symbol(z,z,labels,mask);metrics=enriched_scores(z,z,labels,mask)
  self.assertAlmostEqual(losses[:,1].mean(),metrics['ber_upper'])
  delta=np.tile([.1,-.2],(114,1));result=paired_blocks(delta,32,reps=100)
  np.testing.assert_allclose(result['CI95'],[[.1,.1],[-.2,-.2]],atol=1e-14)
if __name__=='__main__':unittest.main()
