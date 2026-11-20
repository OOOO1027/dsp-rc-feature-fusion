"""Phase 7: front-end training size x soft feedback x local nonlinear branch.

Modes (see runs/phase7_frontend_softfeedback/PROTOCOL.md):
  prepare  hash inputs; verify F=10000 refits reproduce saved Phase-6 front-ends
  smoke    509 training segment only
  develop  10 development captures, train/validation only; selection + gate
  freeze   write frozen.json (only if gate passed)
  confirm  one-shot evaluation on 27 never-scored captures
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
RC = HERE.parent
sys.path.insert(0, str(RC / 'src'))
sys.path.insert(0, str(RC / 'decision_feedback'))
import dfe                                   # noqa: E402  (Phase 6, import only)
import experiment as p6                      # noqa: E402  (Phase 6 helpers, import only)
from dsp_rc.core import symbol_features      # noqa: E402
from dsp_rc.experiment import ChannelPipeline, mask_for  # noqa: E402
from dsp_rc.phase4 import paired_blocks      # noqa: E402

OUT = RC / 'runs/phase7_frontend_softfeedback'
CH = ('rx1', 'rx2')
GUARD = 128
DEV = [509, 513, 517, 519, 523, 528, 529, 533, 535, 382]
DEV_FRESH_VAL = [513, 517, 519, 523, 528, 533, 382]
CONF = [464, 465, 468, 471, 473, 477, 478, 482, 487,
        422, 424, 433, 435, 439, 445, 447, 451, 453,
        542, 545, 547, 551, 555, 560, 563, 565, 569]
FS = (10000, 40000)
CANDS = [dict(name=f'F{F // 1000}k_{B}_{"poly" if N else "nopoly"}', F=F, B=B, N=N, on=int(F == 40000) + int(B != 'hard') + int(N))
         for F, B, N in itertools.product(FS, ['hard', 'posterior_mean'], [False, True])]
PHASE6 = 'F10k_hard_nopoly'
TX = p6.NROOT / 'txdata/RF0.105_data36QAM.txt'


# ---------------------------------------------------------------- features
def posterior_mean(z, s2):
    def pm(x):
        a = -(x[:, None] - dfe.LEVELS[None, :]) ** 2 / (2 * s2)
        a -= a.max(axis=1, keepdims=True)
        w = np.exp(a)
        return (w * dfe.LEVELS).sum(1) / w.sum(1)
    return pm(np.real(z)) + 1j * pm(np.imag(z))


def poly_branch(z, scale):
    c = symbol_features(z, 3) / scale                    # 6 reals: I/Q of t-1, t, t+1
    iu = np.triu_indices(6)
    return np.column_stack([(c[:, :, None] * c[:, None, :])[:, iu[0], iu[1]], c ** 3])   # 21 + 6


# ---------------------------------------------------------------- data
class Guard:
    """Records every capture whose RX file is opened; develop forbids confirmation captures."""
    def __init__(self, allowed):
        self.allowed, self.opened = set(allowed), []

    def check(self, index):
        if index not in self.allowed:
            raise PermissionError(f'capture {index} is not allowed in this mode')
        self.opened.append(index)


def load_raw(index, guard, label_limit):
    guard.check(index)
    audit = p6.audit_hashes()
    paths = p6.raw_paths(index)
    for k, p in list(paths.items()) + [('tx', TX)]:
        if audit.get(str(p)) != p6.sha(p):
            raise RuntimeError(f'hash mismatch {p}')
    rx = {ch: np.loadtxt(paths[ch]).ravel() for ch in CH}
    tx = p6.first_tokens(TX, 40000)                       # only training-segment Tx samples
    if label_limit is None:
        lab = {k: np.loadtxt(paths[k]).ravel() for k in ['i', 'q', 'pam_i', 'pam_q']}
    else:
        lab = {k: p6.first_tokens(paths[k], label_limit) for k in ['i', 'q', 'pam_i', 'pam_q']}
    return rx, tx, lab['i'] + 1j * lab['q'], np.column_stack([lab['pam_i'], lab['pam_q']]).astype(int)


def fit_frontends(index, F, rx, tx, y_tr):
    c = p6.config_for(index)
    c['frontend_train_samples'] = F
    pipes, info = {}, {}
    for ch in CH:
        try:
            p = ChannelPipeline(ch, c).fit(rx[ch][:40000], tx, y_tr)
        except FloatingPointError as e:
            return None, {'diverged': True, 'channel': ch, 'error': str(e)}
        if not (np.isfinite(p.frontend.W).all() and np.isfinite(p.frontend.V).all()):
            return None, {'diverged': True, 'channel': ch}
        pipes[ch] = p
        info[ch] = {'frozen_hash': p.frozen_hash(), 'alignment': p.alignment,
                    'best_train_symbol_lag': int(max(p.symbol_alignment_diagnostic, key=p.symbol_alignment_diagnostic.get))}
    return pipes, info


def segments(pipes, rx, y, labels, splits):
    d = {}
    for split, (a, b) in splits.items():
        e = {'z': {}, 'lms': {}, 'y': y[a:b], 'labels': labels[a:b], 'mask': mask_for(b - a, GUARD)}
        for ch in CH:
            z = pipes[ch].preprocess(rx[ch][4 * a:4 * b], a)[1]
            e['z'][ch] = z
            e['lms'][ch] = pipes[ch].lms.predict(z) * pipes[ch].target_scale
        d[split] = e
    return d


# ---------------------------------------------------------------- models
def predict_family(tr, ev, wanted, causal=False):
    """Predictions (XY and per channel) for one front-end level.

    wanted: list of (B, N) pairs; plus 'ridge' always. causal=True gives K=8 past.
    """
    m = tr['mask']
    out = {'ridge': {}}
    for B, N in wanted:
        out[(B, N)] = {}
    for ch in CH:
        win = dfe.Window(tr['z'][ch], m)
        Ut, Ue = win(tr['z'][ch]), win(ev['z'][ch])
        W0 = dfe.ridge_fit(dfe.with_bias(Ut), tr['y'], m)
        p_ev = dfe.ridge_apply(dfe.with_bias(Ue), W0)
        out['ridge'][ch] = p_ev
        p_cf = dfe.crossfit_first_pass(Ut, tr['y'], m)
        s2 = float(np.mean(np.r_[(p_cf.real - tr['y'].real)[m], (p_cf.imag - tr['y'].imag)[m]] ** 2))
        polys = (poly_branch(tr['z'][ch], win.scale), poly_branch(ev['z'][ch], win.scale))
        K, span = (8, 'past') if causal else (16, 'past+future')
        for B, N in wanted:
            f = dfe.hard if B == 'hard' else (lambda z: posterior_mean(z, s2))
            bt, be = dfe.feedback_taps(f(p_cf), K, span), dfe.feedback_taps(f(p_ev), K, span)
            Ft = dfe.with_bias(Ut, bt, *( [polys[0]] if N else []))
            Fe = dfe.with_bias(Ue, be, *( [polys[1]] if N else []))
            out[(B, N)][ch] = dfe.ridge_apply(Fe, dfe.ridge_fit(Ft, tr['y'], m))
    for v in out.values():
        v['xy'] = (v['rx1'] + v['rx2']) / 2
    return out


def score_xy(pred, ev, per_channel=False):
    outs = ['rx1', 'rx2', 'xy'] if per_channel else ['xy']
    res, loss = {}, {}
    for o in outs:
        res[o], l = dfe.score(pred[o], ev['y'], ev['labels'], ev['mask'])
        loss[o] = l['viterbi_ber']
    return res, loss


# ---------------------------------------------------------------- workers
def run_capture(index, mode, selected=None):
    t0 = time.perf_counter()
    allowed = DEV if mode in ('develop', 'prepare', 'smoke') else CONF
    guard = Guard(allowed)
    if mode == 'smoke':
        rx, tx, y, labels = load_raw(index, guard, 10000)
        splits = {'train': (0, 6000), 'eval': (6000, 10000)}
        y_tr = y[:6000]
    elif mode == 'develop':
        rx, tx, y, labels = load_raw(index, guard, 20000)
        splits = {'train': (0, 10000), 'eval': (10000, 20000)}
        y_tr = y[:10000]
    else:
        rx, tx, y, labels = load_raw(index, guard, None)
        splits = {'train': (0, 10000), 'eval': (20000, 51200)}
        y_tr = y[:10000]
    r = {'index': index, 'frontends': {}, 'cells': {}, 'refs': {}, 'excluded': None, 'diverged': []}
    fronts = {}
    for F in FS:
        if mode == 'smoke':
            c = p6.config_for(index); c['frontend_train_samples'] = min(F, 24000)
            pipes = {ch: ChannelPipeline(ch, c).fit(rx[ch][:24000], tx[:24000], y_tr) for ch in CH}
            info = {}
        else:
            pipes, info = fit_frontends(index, F, rx, tx, y_tr)
        r['frontends'][str(F)] = info
        if pipes is None:
            r['diverged'].append(F)
        fronts[F] = pipes
    if fronts[10000] is not None and mode != 'smoke':
        lags = [r['frontends']['10000'][ch]['best_train_symbol_lag'] for ch in CH]
        if any(l != 0 for l in lags):
            r['excluded'] = f'best training symbol lag {lags} != 0'
            return r
    wanted = [(B, N) for B in ('hard', 'posterior_mean') for N in (False, True)]
    losses = {}
    for F in FS:
        if fronts[F] is None:
            continue
        seg = segments(fronts[F], rx, y, labels, splits)
        tr, ev = seg['train'], seg['eval']
        pred = predict_family(tr, ev, wanted)
        for (B, N) in wanted:
            name = f'F{F // 1000}k_{B}_{"poly" if N else "nopoly"}'
            full = mode == 'develop' or name in (PHASE6, (selected or {}).get('name'))
            r['cells'][name], l = score_xy(pred[(B, N)], ev, per_channel=full)
            losses[name] = l['xy']
        res, l = score_xy(pred['ridge'], ev)
        r['refs'][f'ridge_F{F // 1000}k'] = res
        losses[f'ridge_F{F // 1000}k'] = l['xy']
        if F == 10000:
            lms = {ch: ev['lms'][ch] for ch in CH}; lms['xy'] = (lms['rx1'] + lms['rx2']) / 2
            r['refs']['lms_F10k'] = score_xy(lms, ev)[0]
        if mode == 'confirm' and selected and selected['F'] == F:
            pc = predict_family(tr, ev, [(selected['B'], selected['N'])], causal=True)
            r['refs']['selected_causal_K8_past'] = score_xy(pc[(selected['B'], selected['N'])], ev)[0]
        r.setdefault('first_pass_decision_error_rate', {})[str(F)] = float(np.mean(np.r_[
            [dfe.hard(pred['ridge']['xy'])[ev['mask']].real != ev['y'][ev['mask']].real,
             dfe.hard(pred['ridge']['xy'])[ev['mask']].imag != ev['y'][ev['mask']].imag]]))
    if mode == 'confirm' and selected and selected['name'] in losses and PHASE6 in losses:
        d = losses[selected['name']] - losses[PHASE6]
        r['bootstrap_selected_vs_phase6'] = paired_blocks(d[:, None], 512, 2000, 404)
    r['opened_captures'] = guard.opened
    r['seconds'] = time.perf_counter() - t0
    return r


def pool_map(indices, mode, selected=None, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {i: ex.submit(run_capture, i, mode, selected) for i in indices}
        res = {}
        for i, f in futs.items():
            res[str(i)] = f.result()
            print(i, 'done', f"{res[str(i)].get('seconds', 0):.0f}s", res[str(i)]['excluded'] or '', flush=True)
    return res


# ---------------------------------------------------------------- modes
def check_protocol():
    fz = json.loads((OUT / 'protocol_freeze.json').read_text())
    assert p6.sha(OUT / 'PROTOCOL.md') == fz['PROTOCOL.md'] and p6.sha(OUT / 'protocol.json') == fz['protocol.json']
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['status'] == 'frozen' and len(p['candidates']) == 8
    assert [c['name'] for c in p['candidates']] == [c['name'] for c in CANDS]
    return p


def clean(x):
    """JSON-safe: non-finite floats (diverged = counted as loss) become None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write(name, data):
    p6.write_json(OUT / name, clean(data))


def prepare():
    check_protocol()
    rows = {}
    for index in DEV:
        guard = Guard(DEV)
        rx, tx, y, labels = load_raw(index, guard, 10000)
        pipes, info = fit_frontends(index, 10000, rx, tx, y[:10000])
        saved = {ch: ChannelPipeline.load(p6.frontend_path(index, ch), ch, p6.config_for(index)).frozen_hash() for ch in CH}
        ok = all(info[ch]['frozen_hash'] == saved[ch] for ch in CH)
        rows[str(index)] = {'refit_matches_saved_frontend': ok, 'saved': saved, 'refit': {ch: info[ch]['frozen_hash'] for ch in CH}}
        print(index, 'F=10000 refit reproduces saved front-end:', ok, flush=True)
        if not ok:
            write('prepare.json', rows)
            raise SystemExit(f'Protocol stop: F=10000 refit does not reproduce saved front-end for {index}')
    audit = p6.audit_hashes()
    manifest = {str(i): {k: {'path': str(p), 'sha256': audit[str(p)]} for k, p in p6.raw_paths(i).items()} for i in DEV + CONF}
    write('prepare.json', rows)
    write('input_manifest.json', manifest)
    write('environment.json', p6.environment())


def smoke():
    check_protocol()
    r = run_capture(509, 'smoke')
    write('smoke.json', {k: v for k, v in r.items()})
    for k, v in r['cells'].items():
        print(f"{k:28s} XY viterbi {v['xy']['viterbi_ber']:.5f}")


def develop():
    check_protocol()
    assert not (OUT / 'develop_results.json').exists()
    res = pool_map(DEV, 'develop')
    assert all(set(r['opened_captures']) <= set(DEV) for r in res.values())
    table, logr = [], {}
    for c in CANDS:
        lr = {}
        for i, r in res.items():
            if r['excluded']:
                continue
            base = r['cells'][PHASE6]['xy']['viterbi_ber']
            lr[i] = (np.log(r['cells'][c['name']]['xy']['viterbi_ber'] / base) if c['name'] in r['cells'] else np.inf)
        logr[c['name']] = lr
        vals = np.array(list(lr.values()))
        fresh = np.array([lr[str(i)] for i in DEV_FRESH_VAL if str(i) in lr])
        table.append({'name': c['name'], 'on': c['on'], 'F': c['F'], 'B': c['B'], 'N': c['N'],
                      'median_log_ratio': float(np.median(vals)), 'median_rel_change': float(np.expm1(np.median(vals))),
                      'wins': int(np.sum(vals < 0)), 'n': int(len(vals)),
                      'fresh_median_rel_change': float(np.expm1(np.median(fresh)))})
    pref = {'F': 0, 'B': 1, 'N': 2}
    order = sorted(table, key=lambda t: (t['median_log_ratio'], t['on'], (t['F'] == 40000) * 4 + (t['B'] != 'hard') * 2 + int(t['N'])))
    sel = next(c for c in CANDS if c['name'] == order[0]['name'])
    top = order[0]
    gate = {'a_median_rel_change': top['median_rel_change'], 'a_pass': top['median_rel_change'] <= -0.05,
            'b_wins': top['wins'], 'b_pass': top['wins'] >= 7,
            'c_fresh_median_rel_change': top['fresh_median_rel_change'], 'c_pass': top['fresh_median_rel_change'] < 0}
    gate['pass'] = gate['a_pass'] and gate['b_pass'] and gate['c_pass'] and sel['name'] != PHASE6
    write('develop_results.json', res)
    write('selection.json', {'ranking': order, 'selected': sel, 'log_ratios': logr})
    write('decision.json', {'gate': gate, 'proceed_to_confirmation': gate['pass'], 'confirmation_evaluated': False})
    for t in order:
        print(f"{t['name']:28s} median {100 * t['median_rel_change']:+6.1f}%  wins {t['wins']}/{t['n']}  fresh-val median {100 * t['fresh_median_rel_change']:+6.1f}%")
    print('selected', sel['name'], 'gate', gate)


def code_hashes():
    files = [HERE / 'p7.py', HERE / 'test_p7.py', RC / 'decision_feedback/dfe.py', RC / 'decision_feedback/experiment.py'] + sorted((RC / 'src/dsp_rc').glob('*.py'))
    return {str(f.relative_to(p6.ROOT)): p6.sha(f) for f in files}


def freeze():
    check_protocol()
    d = json.loads((OUT / 'decision.json').read_text())
    if not d['proceed_to_confirmation']:
        raise SystemExit('Gate failed: protocol forbids reading confirmation data.')
    s = json.loads((OUT / 'selection.json').read_text())
    write('frozen.json', {'utc': p6.now(), 'selected': s['selected'], 'code_sha256': code_hashes(),
                          'develop_results_sha256': p6.sha(OUT / 'develop_results.json'),
                          'protocol_freeze': json.loads((OUT / 'protocol_freeze.json').read_text())})
    print('frozen', s['selected']['name'])


def rel(a, b):
    return a / b - 1


def confirm():
    check_protocol()
    fz = json.loads((OUT / 'frozen.json').read_text())
    assert fz['code_sha256'] == code_hashes(), 'code changed after freeze'
    assert not (OUT / 'confirm_results.json').exists()
    sel = fz['selected']
    res = pool_map(CONF, 'confirm', sel)
    v = lambda r, name: (r['cells'][name]['xy']['viterbi_ber'] if name in r['cells'] else None)
    counted, rows = [], []
    for i in map(str, CONF):
        r = res[i]
        row = {'index': int(i), 'excluded': r['excluded'], 'diverged': r['diverged']}
        if not r['excluded']:
            base = v(r, PHASE6)
            row.update({'phase6': base, 'selected': v(r, sel['name']),
                        'ridge_F10k_viterbi': r['refs']['ridge_F10k']['xy']['viterbi_ber'],
                        'ridge_F10k_sbs_lower': r['refs']['ridge_F10k']['xy']['ber_lower'],
                        'ridge_F40k_viterbi': r['refs'].get('ridge_F40k', {}).get('xy', {}).get('viterbi_ber'),
                        'lms_sbs_lower': r['refs']['lms_F10k']['xy']['ber_lower'],
                        'selected_causal': r['refs'].get('selected_causal_K8_past', {}).get('xy', {}).get('viterbi_ber'),
                        'phase6_bit_errors': int(round(base * 4 * r['cells'][PHASE6]['xy']['symbols'])) if base is not None else None,
                        'ci512_selected_vs_phase6': r.get('bootstrap_selected_vs_phase6', {}).get('CI95', [[None, None]])[0],
                        'first_pass_error_F10k': r.get('first_pass_decision_error_rate', {}).get('10000')})
            if base is not None and row['phase6_bit_errors'] >= 30:
                counted.append(r)
        rows.append(row)

    def contrast(a_name, b_name, rs):
        ch = []
        for r in rs:
            a, b = v(r, a_name), v(r, b_name)
            ch.append(np.inf if a is None else rel(a, b))   # diverged candidate counts as a loss
        ch = np.array(ch)
        wins = int(np.sum(ch < 0)); n = len(ch)
        return {'wins': wins, 'n': n, 'median_rel_change': float(np.median(ch)),
                'sign_test_p': float(binomtest(wins, n, 0.5).pvalue) if n else None, 'changes': ch.tolist()}

    h2 = contrast(sel['name'], PHASE6, counted)
    h2['supported'] = h2['wins'] / h2['n'] >= 0.8 and h2['median_rel_change'] <= -0.10 and h2['sign_test_p'] < 0.01
    h1c = []
    for r in counted:
        a = r['refs'].get('ridge_F40k', {}).get('xy', {}).get('viterbi_ber')
        h1c.append(np.inf if a is None else rel(a, r['refs']['ridge_F10k']['xy']['viterbi_ber']))
    h1c = np.array(h1c)
    h1 = {'wins': int(np.sum(h1c < 0)), 'n': len(h1c), 'median_rel_change': float(np.median(h1c))}
    h1['supported'] = h1['wins'] / h1['n'] >= 0.8 and h1['median_rel_change'] <= -0.05

    def factor(kind):
        pairs = {}
        for F in FS:
            for other in (['nopoly', 'poly'] if kind == 'B' else ['hard', 'posterior_mean']):
                if kind == 'B':
                    a, b = f'F{F // 1000}k_posterior_mean_{other}', f'F{F // 1000}k_hard_{other}'
                else:
                    a, b = f'F{F // 1000}k_{other}_poly', f'F{F // 1000}k_{other}_nopoly'
                pairs[f'{a} vs {b}'] = contrast(a, b, counted)
        pooled = np.concatenate([p['changes'] for p in pairs.values()])
        ok = sum(p['median_rel_change'] < 0 and p['sign_test_p'] < 0.05 for p in pairs.values())
        return {'pairs': pairs, 'pooled_median_rel_change': float(np.median(pooled)), 'pairs_supported': ok,
                'supported': bool(np.median(pooled) < 0 and ok >= 3)}

    h3, h4 = factor('B'), factor('N')
    write('confirm_results.json', {'utc': p6.now(), 'selected': sel, 'per_capture': res, 'rows': rows,
                                   'counted_captures': [r['index'] for r in counted],
                                   'H1': h1, 'H2': h2, 'H3': h3, 'H4': h4})
    for row in rows:
        print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in row.items() if k not in ('ci512_selected_vs_phase6',)})
    print('H1', {k: h1[k] for k in ['wins', 'n', 'median_rel_change', 'supported']})
    print('H2', {k: h2[k] for k in ['wins', 'n', 'median_rel_change', 'sign_test_p', 'supported']})
    print('H3', h3['pooled_median_rel_change'], h3['pairs_supported'], h3['supported'])
    print('H4', h4['pooled_median_rel_change'], h4['pairs_supported'], h4['supported'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'smoke', 'develop', 'freeze', 'confirm'])
    {'prepare': prepare, 'smoke': smoke, 'develop': develop, 'freeze': freeze, 'confirm': confirm}[ap.parse_args().mode]()


if __name__ == '__main__':
    main()
