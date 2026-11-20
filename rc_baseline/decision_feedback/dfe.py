"""Phase 6 numerical components: decisions, feedback features, Viterbi, scoring.

Pure functions on arrays; nothing here reads files or labels by itself.
"""
import numpy as np

from dsp_rc.core import symbol_features, fit_ridge, ber_metrics
from dsp_rc.fusion import LinearReservoir
from dsp_rc.core import Reservoir

LEVELS = np.arange(6) * 2.0 - 5.0          # PAM6 levels -5..5, index q = 0..5
POPCOUNT = np.array([bin(i).count('1') for i in range(8)])


def hard(z):
    """Nearest PAM6 level per I/Q component (first-pass decision)."""
    d = lambda x: np.clip(2 * np.round((x + 5) / 2) - 5, -5, 5)
    return d(np.real(z)) + 1j * d(np.imag(z))


def shifted(sym, k):
    """out[t] = sym[t-k] (k>0 past, k<0 future); zero outside the segment."""
    out = np.zeros_like(sym)
    if k > 0:
        out[k:] = sym[:-k]
    elif k < 0:
        out[:k] = sym[-k:]
    else:
        out[:] = sym
    return out


def feedback_taps(dec, K, span):
    ks = list(range(1, K + 1)) + ([-k for k in range(1, K + 1)] if span == 'past+future' else [])
    cols = []
    for k in ks:
        s = shifted(dec, k) / 5.0
        cols += [s.real, s.imag]
    return np.column_stack(cols)


def feedback_reservoir(dec, activation, seed, nodes=40):
    """Causal reservoir over past decisions: state at t sees d_0..d_{t-1}."""
    x = np.column_stack([shifted(dec, 1).real, shifted(dec, 1).imag]) / 5.0
    if activation == 'identity':
        r = LinearReservoir(2, nodes, leak=.7, norm_bound=.6, seed=seed)
    else:
        r = Reservoir(2, nodes, leak=.7, spectral_radius=.9, seed=seed)
    states, _, _ = r.transform(x)
    return states


class Window:
    """115-symbol interleaved I/Q window scaled by training max-abs (as ridge_low)."""
    def __init__(self, z_train, mask, taps=115):
        self.taps = taps
        self.scale = float(np.max(np.abs(symbol_features(z_train, taps)[mask])))

    def __call__(self, z):
        return symbol_features(z, self.taps) / self.scale


def ridge_fit(F, y, mask, lam=1e-5):
    return fit_ridge(F[mask], np.column_stack([y.real, y.imag])[mask], lam)


def ridge_apply(F, W):
    o = F @ W
    return o[:, 0] + 1j * o[:, 1]


def with_bias(*blocks):
    n = len(blocks[0])
    return np.column_stack(list(blocks) + [np.ones(n)])


def crossfit_first_pass(U_train, y_train, mask, folds=5, guard=128, lam=1e-5):
    """Out-of-fold first-pass predictions on the training segment.

    Block f's predictions come from a ridge fitted on mask positions outside
    [start-guard, end+guard); block f's labels are never used for block f.
    """
    n = len(y_train)
    edges = np.linspace(0, n, folds + 1).astype(int)
    F = with_bias(U_train)
    pred = np.empty(n, dtype=complex)
    for f in range(folds):
        a, b = edges[f], edges[f + 1]
        fit_mask = mask.copy()
        fit_mask[max(0, a - guard):min(n, b + guard)] = False
        W = ridge_fit(F, y_train, fit_mask, lam)
        pred[a:b] = ridge_apply(F[a:b], W)
    return pred


# ---------------------------------------------------------------- Viterbi
def viterbi_component(x):
    """3-state NLTCP trellis on one real component.

    State s = floor(q/2) in {0,1,2}; allowed q_t = s+m, m=0..3; next state
    floor(q_t/2). Euclidean branch metric, unknown initial state (all zero).
    Returns PAM6 indices q (length T).
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    # transitions: from s, q in s..s+3
    from_s = np.repeat(np.arange(3), 4)
    q_of = from_s + np.tile(np.arange(4), 3)
    to_s = q_of // 2
    cost = np.zeros(3)
    back_s = np.zeros((T, 3), dtype=np.int8)
    back_q = np.zeros((T, 3), dtype=np.int8)
    branch = (x[:, None] - LEVELS[None, :]) ** 2          # (T,6)
    for t in range(T):
        cand = cost[from_s] + branch[t, q_of]               # 12 transitions
        new = np.full(3, np.inf)
        bs = np.zeros(3, dtype=np.int8)
        bq = np.zeros(3, dtype=np.int8)
        for j in range(12):                                  # first minimum wins ties
            ns = to_s[j]
            if cand[j] < new[ns]:
                new[ns] = cand[j]; bs[ns] = from_s[j]; bq[ns] = q_of[j]
        cost = new - new.min()
        back_s[t] = bs; back_q[t] = bq
    s = int(np.argmin(cost))
    q = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        q[t] = back_q[t, s]; s = back_s[t, s]
    return q


def viterbi_messages(z):
    """Decoded PAM4 messages (T,2); entry 0 has no predecessor and is unscored."""
    q = np.column_stack([viterbi_component(np.real(z)), viterbi_component(np.imag(z))])
    m = np.zeros_like(q)
    m[1:] = q[1:] - q[:-1] // 2
    assert m[1:].min() >= 0 and m[1:].max() <= 3, 'Viterbi produced an illegal message'
    return m


# ---------------------------------------------------------------- scoring
def score(z, y, labels, mask):
    """All protocol metrics plus per-symbol losses for paired bootstrap."""
    sbs = ber_metrics(z, labels, mask)
    m = viterbi_messages(z)
    lab = labels[mask]
    v_err = POPCOUNT[np.bitwise_xor(m[mask], lab)].sum(axis=1)
    from dsp_rc.core import decode_nltcp
    _, d = decode_nltcp(z)
    d = d[mask]
    valid = d <= 3
    s_err = POPCOUNT[np.bitwise_xor(np.minimum(d, 7), lab)]
    s_lower = np.where(valid, s_err, 0).sum(axis=1)
    s_upper = np.where(valid, s_err, 2).sum(axis=1)
    n = int(mask.sum())
    metrics = {
        'mse': float(np.mean(np.abs(z[mask] - y[mask]) ** 2)),
        'ber_lower': sbs['ber_lower'], 'ber_upper': sbs['ber_upper'],
        'invalid_rate': sbs['invalid_pam4_decisions'] / (2 * n),
        'auto_ber': sbs['matlab_compatible_ber'],
        'pam4_pair_error_rate_sbs': float(np.mean(np.any(np.where(valid, d, -1) != lab, axis=1))),
        'viterbi_ber': float(v_err.sum() / (4 * n)),
        'viterbi_pam4_pair_error_rate': float(np.mean(v_err > 0)),
        'symbols': n,
    }
    losses = {'mse': np.abs(z[mask] - y[mask]) ** 2, 'viterbi_ber': v_err / 4.0,
              'ber_lower': s_lower / 4.0, 'ber_upper': s_upper / 4.0}
    return metrics, losses


def error_runs(per_symbol_err):
    """Lengths of consecutive erroneous symbols."""
    e = np.asarray(per_symbol_err) > 0
    runs, cur = [], 0
    for v in e:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = np.array(runs, dtype=int)
    if runs.size == 0:
        return {'count': 0}
    return {'count': int(runs.size), 'median': float(np.median(runs)), 'max': int(runs.max()),
            'share_len_ge3': float(np.mean(runs >= 3))}
