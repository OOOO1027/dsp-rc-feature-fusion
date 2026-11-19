"""Synthetic-data tests for Phase 6 (no real data, no labels from files)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
sys.path.insert(0, str(HERE))
import dfe  # noqa: E402
from experiment import first_tokens  # noqa: E402
from dsp_rc.core import ber_metrics  # noqa: E402
from dsp_rc.experiment import mask_for  # noqa: E402


def encode(m, q0=0):
    """NLTCP encoder q_t = floor(q_{t-1}/2) + m_t, returns PAM6 levels."""
    q = np.empty(len(m), dtype=int)
    prev = q0
    for t, v in enumerate(m):
        q[t] = prev // 2 + v
        prev = q[t]
    return dfe.LEVELS[q]


class TestViterbi(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.m = rng.integers(0, 4, (3000, 2))
        self.y = encode(self.m[:, 0]) + 1j * encode(self.m[:, 1])
        self.labels = self.m.copy()
        self.mask = mask_for(3000, 128)

    def test_noiseless_is_error_free(self):
        dec = dfe.viterbi_messages(self.y)
        np.testing.assert_array_equal(dec[1:], self.m[1:])

    def test_never_illegal_and_not_worse_than_symbolwise(self):
        rng = np.random.default_rng(1)
        z = self.y + 0.55 * (rng.standard_normal(3000) + 1j * rng.standard_normal(3000))
        met, loss = dfe.score(z, self.y, self.labels, self.mask)
        self.assertLessEqual(met['viterbi_ber'], met['ber_lower'] + 1e-12)
        ref = ber_metrics(z, self.labels, self.mask)
        self.assertAlmostEqual(loss['ber_lower'].mean(), ref['ber_lower'], places=12)
        self.assertAlmostEqual(loss['ber_upper'].mean(), ref['ber_upper'], places=12)
        self.assertAlmostEqual(loss['viterbi_ber'].mean(), met['viterbi_ber'], places=12)


class TestFeatures(unittest.TestCase):
    def test_shift_directions(self):
        s = np.arange(1, 6) + 0j
        np.testing.assert_array_equal(dfe.shifted(s, 1).real, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(dfe.shifted(s, -2).real, [3, 4, 5, 0, 0])
        F = dfe.feedback_taps(s, 2, 'past+future')
        self.assertEqual(F.shape, (5, 8))

    def test_reservoir_feedback_is_causal(self):
        rng = np.random.default_rng(2)
        d = dfe.hard(rng.normal(0, 3, 400) + 1j * rng.normal(0, 3, 400))
        a = dfe.feedback_reservoir(d, 'identity', 200)
        d2 = d.copy(); d2[300] += 2
        b = dfe.feedback_reservoir(d2, 'identity', 200)
        np.testing.assert_array_equal(a[:301], b[:301])   # state at t<=300 sees d_<t only
        self.assertFalse(np.allclose(a[301:], b[301:]))

    def test_crossfit_block_ignores_own_labels(self):
        rng = np.random.default_rng(3)
        n = 2000
        U = rng.standard_normal((n, 6))
        y = U @ rng.standard_normal(6) + 1j * (U @ rng.standard_normal(6))
        mask = mask_for(n, 128)
        p = dfe.crossfit_first_pass(U, y, mask, folds=5, guard=128)
        y2 = y.copy(); y2[800:1200] += 50          # fold 2 is [800,1200)
        p2 = dfe.crossfit_first_pass(U, y2, mask, folds=5, guard=128)
        np.testing.assert_allclose(p[800:1200], p2[800:1200])
        self.assertFalse(np.allclose(p[:400], p2[:400]))


class TestLabelReader(unittest.TestCase):
    def test_first_tokens_matches_prefix(self):
        vals = np.random.default_rng(4).integers(0, 4, 200000).astype(float)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.txt'
            p.write_text('  '.join(f'{v:.7e}' for v in vals))     # single line, like PAM4 files
            got = first_tokens(p, 20000)
        np.testing.assert_array_equal(got, vals[:20000])


if __name__ == '__main__':
    unittest.main()
