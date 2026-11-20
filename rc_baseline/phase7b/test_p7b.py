"""Phase 7b tests: guard rule (synthetic + 382 training segment), fallback, access."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7b  # noqa: E402
import p7   # noqa: E402


class TestGuard(unittest.TestCase):
    def test_fires_on_diverge_then_recover(self):
        rng = np.random.default_rng(0)
        e = rng.normal(0, 0.25, 40000)
        e[15000:25000] *= 3000          # transient blow-up, finite afterwards
        self.assertGreater(p7b.guard_ratio(e), 10)

    def test_quiet_on_stationary_and_converging(self):
        rng = np.random.default_rng(1)
        e = rng.normal(0, 1, 40000) * np.linspace(1.5, 0.8, 40000)
        self.assertLess(p7b.guard_ratio(e), 10)

    def test_nonfinite_is_infinite_ratio(self):
        e = np.ones(40000); e[100] = np.nan
        self.assertEqual(p7b.guard_ratio(e), np.inf)


class TestReal382(unittest.TestCase):
    def test_382_rx1_fires_rx2_not_and_fallback_hash(self):
        g = p7.Guard(p7.DEV)
        rx, tx, y, _ = p7.load_raw(382, g, 10000)
        fronts, info = p7b.fit_levels(382, rx, tx, y[:10000])
        self.assertTrue(info['F40g']['rx1']['guard_fired'])
        self.assertFalse(info['F40g']['rx2']['guard_fired'])
        self.assertEqual(info['F40g']['rx1']['frozen_hash'], info['F10k']['rx1']['frozen_hash'])
        self.assertNotEqual(info['F40g']['rx2']['frozen_hash'], info['F10k']['rx2']['frozen_hash'])


class TestAccess(unittest.TestCase):
    def test_develop_cannot_open_confirmation_or_reserve(self):
        for idx in (464, 583, 390):
            g = p7.Guard(p7b.allowed_for('develop'))
            with self.assertRaises(PermissionError):
                p7.load_raw(idx, g, 20000)

    def test_confirm_cannot_open_reserve(self):
        g = p7.Guard(p7b.allowed_for('confirm'))
        with self.assertRaises(PermissionError):
            p7.load_raw(583, g, None)

    def test_sets_disjoint(self):
        self.assertFalse(set(p7b.DEV) & set(p7b.CONF) or set(p7b.RESERVE) & (set(p7b.DEV) | set(p7b.CONF)))
        self.assertEqual(len(p7b.CANDS), 8)


if __name__ == '__main__':
    unittest.main()
