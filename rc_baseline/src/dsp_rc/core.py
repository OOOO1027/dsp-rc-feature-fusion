"""Numerical modules. Fit methods accept training data only.

Rows are time samples; MATLAB's O matrix is the transpose of our features.
All predictive transforms are independent of held-out labels.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import correlate, correlation_lags


def rms(x):
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def align_parameters(rx_train, tx_train, max_lag=64):
    x = rx_train - np.mean(rx_train)
    y = tx_train - np.mean(tx_train)
    c = correlate(x, y, mode='full', method='fft')
    lags = correlation_lags(len(x), len(y))
    keep = np.abs(lags) <= max_lag
    k = np.argmax(np.abs(c[keep]))
    lag = int(lags[keep][k]); peak = float(c[keep][k])
    return {'lag': lag, 'polarity': 1 if peak >= 0 else -1,
            'training_correlation': peak / np.sqrt(np.sum(x*x)*np.sum(y*y))}


def align_segment(x, lag, polarity):
    """y[t]=polarity*x[t+lag], zero outside this split (never wrap)."""
    y = np.zeros_like(x)
    if lag >= 0:
        y[:len(x)-lag] = polarity*x[lag:]
    else:
        y[-lag:] = polarity*x[:len(x)+lag]
    return y


def windows(x, taps):
    if taps % 2 != 1:
        raise ValueError('Odd centered window required')
    return sliding_window_view(np.pad(x, (taps//2, taps//2)), taps)


def symbol_features(z, taps=25):
    """Chronological window, interleaved [I(t-h),Q(t-h),...,I(t+h),Q(t+h)]."""
    w = windows(z, taps)
    return np.stack([w.real, w.imag], axis=-1).reshape(len(z), 2*taps)


def rrc(rolloff=0.105, span=10, sps=4):
    """Unit-energy root raised cosine, matching rcosdesign(beta,span,sps,'sqrt')."""
    t = np.arange(-span*sps//2, span*sps//2+1, dtype=float)/sps
    b = rolloff; h = np.empty_like(t)
    zero = np.isclose(t, 0); singular = np.isclose(np.abs(t), 1/(4*b))
    normal = ~(zero | singular); u = t[normal]
    h[normal] = (np.sin(np.pi*u*(1-b))+4*b*u*np.cos(np.pi*u*(1+b)))/(np.pi*u*(1-(4*b*u)**2))
    h[zero] = 1+b*(4/np.pi-1)
    h[singular] = b/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*b))+(1-2/np.pi)*np.cos(np.pi/(4*b)))
    return h/np.linalg.norm(h)


def cap_demodulate(x, sample_start=0, sps=4, rolloff=0.105, span=10, phase=0):
    carrier = ((1+rolloff)/2+0.01)
    t = np.arange(len(x))+sample_start
    mixed = x*np.exp(-1j*carrier*2*np.pi*t/sps)
    return np.convolve(mixed, rrc(rolloff, span, sps), 'same')[phase::sps]


@dataclass
class VolterraLMS:
    taps: int
    step: float
    quadratic_taps: int
    quadratic_step: float

    def fit(self, x, d):
        """One pass; same reversed windows and triu quadratic basis as MATLAB."""
        h = max(self.taps, self.quadratic_taps)//2
        a, b = self.quadratic_taps, self.taps
        self.W = np.zeros(b); self.V = np.eye(a)*1e-7
        self.errors = []
        for t in range(h, len(x)-h):
            u = x[t-b//2:t+b//2+1][::-1]
            v = x[t-a//2:t+a//2+1][::-1]
            q = np.triu(np.outer(v, v))
            e = d[t]-(self.W@u+np.sum(self.V*q))
            self.W += self.step*e*u
            self.V += self.quadratic_step*e*q
            self.errors.append(e)
        self.errors = np.asarray(self.errors)
        if not np.isfinite(self.W).all() or not np.isfinite(self.V).all():
            raise FloatingPointError('Volterra training diverged')
        return self

    def predict(self, x):
        u = windows(x, self.taps)[:, ::-1]
        v = windows(x, self.quadratic_taps)[:, ::-1]
        # MATLAB sums V .* triu(v v^T); lower triangular weights are irrelevant.
        return u@self.W+np.einsum('ni,ij,nj->n', v, np.triu(self.V), v, optimize=True)


@dataclass
class SymbolLMS:
    taps: int = 115
    step: float = 0.0021

    def fit(self, x, d, mask):
        self.W = np.zeros(self.taps, dtype=complex)
        u = windows(x, self.taps)[:, ::-1]
        self.errors = []
        for t in np.flatnonzero(mask):
            e = d[t]-self.W@u[t]
            self.W += self.step*e*u[t].conj()
            self.errors.append(e)
        self.errors = np.asarray(self.errors)
        if not np.isfinite(self.W).all():
            raise FloatingPointError('Symbol LMS diverged')
        return self

    def predict(self, x):
        return windows(x, self.taps)[:, ::-1]@self.W


@dataclass
class Reservoir:
    input_dim: int
    nodes: int = 80
    leak: float = 0.7
    spectral_radius: float = 0.9
    seed: int = 100

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        a = rng.uniform(-1, 1, (self.nodes, self.nodes))
        self.A = a*(self.spectral_radius/np.max(np.abs(np.linalg.eigvals(a))))
        self.Win = rng.uniform(-1, 1, (self.nodes, self.input_dim+1))

    def transform(self, x, initial_state=None):
        """Fixed A/Win. Returns states, readout features and final state; no labels."""
        state = np.zeros(self.nodes) if initial_state is None else initial_state.copy()
        states = np.empty((len(x), self.nodes))
        drive = np.column_stack([x, np.ones(len(x))])@self.Win.T
        for t in range(len(x)):
            state = (1-self.leak)*state+self.leak*np.tanh(self.A@state+drive[t])
            states[t] = state
        return states, np.column_stack([states, x, np.ones(len(x))]), state.copy()


def fit_ridge(features, target, ridge=1e-5):
    """Readout shape (feature_dim,2); identical ridge objective to MATLAB."""
    gram = features.T@features+ridge*np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T@target)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


def decode_nltcp(z):
    """Original PAM6 nearest-neighbor then one-symbol-memory absolute difference.

    Ties pick the smaller PAM6 level. Raw outputs 4/5 are retained, never clipped.
    First output needs predecessor state and is not scored.
    """
    components = np.column_stack([z.real, z.imag])
    q = np.argmin(abs((components/2+2.5)[..., None]-np.arange(6)), axis=-1)
    d = np.zeros_like(q)
    d[1:] = abs(q[1:]-q[:-1]//2)
    return q, d


def ber_metrics(z, labels, mask):
    """Natural-binary labels. Exact 2-bit error counts on legal decisions.

    Invalid 4/5 are erasures: report [0,2] possible bit errors per erasure,
    not a fabricated 2-bit mapping. Also report MATLAB biterr's auto-width
    per-component convention, which can dilute BER when width expands to 3.
    """
    _, pred = decode_nltcp(z)
    p = pred[mask]; y = labels[mask].astype(int)
    valid = p <= 3
    xor = np.bitwise_xor(p, y)
    popcount = np.array([int(i).bit_count() for i in range(8)])
    legal_errors = int(popcount[xor[valid]].sum())
    erasures = int((~valid).sum()); total_bits = int(y.size*2)
    auto = []
    for c in range(2):
        k = max(1, int(max(p[:, c].max(), y[:, c].max())).bit_length())
        e = int(popcount[xor[:, c]].sum())
        auto.append({'width': k, 'errors': e, 'bits': len(p)*k, 'ber': e/(len(p)*k)})
    return {'symbols': len(p), 'reference_bits': total_bits,
            'legal_bit_errors': legal_errors, 'invalid_pam4_decisions': erasures,
            'bit_errors_lower': legal_errors, 'bit_errors_upper': legal_errors+2*erasures,
            'ber_lower': legal_errors/total_bits,
            'ber_upper': (legal_errors+2*erasures)/total_bits,
            'ber_exact': legal_errors/total_bits if erasures == 0 else None,
            'pam4_ser': float(np.mean(p != y)),
            'matlab_auto_width': auto,
            'matlab_compatible_ber': float(np.mean([a['ber'] for a in auto]))}
