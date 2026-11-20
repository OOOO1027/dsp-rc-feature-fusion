"""Phase 7 tests. Synthetic except the 509 training-only front-end refit check."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7  # noqa: E402
import dfe  # noqa: E402
from dsp_rc.experiment import ChannelPipeline  # noqa: E402


class TestFeatures(unittest.TestCase):
    def test_posterior_mean_tends_to_hard(self):
        rng = np.random.default_rng(0)
        z = rng.uniform(-6, 6, 2000) + 1j * rng.uniform(-6, 6, 2000)
        z = z[(np.abs(((z.real + 5) % 2) - 1) > 0.05) & (np.abs(((z.imag + 5) % 2) - 1) > 0.05)]  # avoid exact boundaries
        np.testing.assert_allclose(p7.posterior_mean(z, 1e-4), dfe.hard(z), atol=1e-9)

    def test_posterior_mean_shrinks_to_center_when_noisy(self):
        self.assertAlmostEqual(abs(p7.posterior_mean(np.array([4.0 + 4j]), 1e6)[0]), 0.0, places=3)

    def test_poly_branch_dimension(self):
        z = np.random.default_rng(1).standard_normal(50) + 1j
        self.assertEqual(p7.poly_branch(z, 1.0).shape, (50, 27))

    def test_candidates_match_protocol(self):
        self.assertEqual(len(p7.CANDS), 8)
        self.assertIn(p7.PHASE6, [c['name'] for c in p7.CANDS])


class TestAccess(unittest.TestCase):
    def test_develop_cannot_open_confirmation_capture(self):
        g = p7.Guard(p7.DEV)
        with self.assertRaises(PermissionError):
            p7.load_raw(464, g, 20000)
        self.assertEqual(g.opened, [])

    def test_dev_and_confirmation_disjoint(self):
        self.assertFalse(set(p7.DEV) & set(p7.CONF))
        self.assertEqual(len(p7.CONF), 27)


class TestRefit(unittest.TestCase):
    def test_509_F10000_refit_reproduces_saved_frontend(self):
        g = p7.Guard(p7.DEV)
        rx, tx, y, _ = p7.load_raw(509, g, 10000)
        pipes, info = p7.fit_frontends(509, 10000, rx, tx, y[:10000])
        for ch in p7.CH:
            saved = ChannelPipeline.load(p7.p6.frontend_path(509, ch), ch, p7.p6.config_for(509)).frozen_hash()
            self.assertEqual(info[ch]['frozen_hash'], saved)


if __name__ == '__main__':
    unittest.main()
