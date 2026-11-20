"""Phase 7b: guarded full-training front-end x soft feedback x local nonlinear branch.

Reuses Phase 7 / Phase 6 code by import only. See runs/phase7b_guarded_frontend/PROTOCOL.md.
Modes: prepare / smoke / develop / freeze / confirm
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
sys.path.insert(0, str(RC / 'phase7'))
import p7                                    # noqa: E402  (import only)
from p7 import dfe, p6                       # noqa: E402
from dsp_rc.experiment import ChannelPipeline  # noqa: E402
from dsp_rc.phase4 import paired_blocks      # noqa: E402

OUT = RC / 'runs/phase7b_guarded_frontend'
CH = p7.CH
DEV, DEV_FRESH_VAL, CONF = p7.DEV, p7.DEV_FRESH_VAL, p7.CONF
RESERVE = [583, 579, 586, 590, 595, 596, 599, 606, 625, 627, 383, 391, 390, 396, 410, 412, 415, 629]
LEVELS = ('F10k', 'F40g')
CANDS = [dict(name=f'{F}_{B}_{"poly" if N else "nopoly"}', F=F, B=B, N=N,
              on=int(F == 'F40g') + int(B != 'hard') + int(N))
         for F, B, N in itertools.product(LEVELS, ['hard', 'posterior_mean'], [False, True])]
PHASE6 = 'F10k_hard_nopoly'


# ---------------------------------------------------------------- stability guard
def guard_ratio(errors, blocks=16):
    """max block MSE / median of first 4 block MSEs of LMS training squared error."""
    e = np.abs(np.asarray(errors)) ** 2
    if not np.isfinite(e).all():
        return np.inf
    B = len(e) // blocks
    b = np.array([e[k * B:(k + 1) * B].mean() for k in range(blocks)])
    return float(b.max() / np.median(b[:4]))


def fit_levels(index, rx, tx, y_tr):
    """Returns {'F10k': pipes or None, 'F40g': pipes or None}, info."""
    info = {'F10k': {}, 'F40g': {}}
    f10, i10 = p7.fit_frontends(index, 10000, rx, tx, y_tr)
    info['F10k'] = i10
    if f10 is None:
        return {'F10k': None, 'F40g': None}, info
    c = p6.config_for(index)
    c['frontend_train_samples'] = 40000
    f40 = {}
    for ch in CH:
        rec = {}
        try:
            p = ChannelPipeline(ch, c).fit(rx[ch][:40000], tx, y_tr)
            finite = np.isfinite(p.frontend.W).all() and np.isfinite(p.frontend.V).all()
            ratio = guard_ratio(p.frontend.errors) if finite else np.inf
            rec.update(ratio=ratio, exception=None)
        except FloatingPointError as e:
            p, ratio = None, np.inf
            rec.update(ratio=None, exception=str(e))
        fired = p is None or not np.isfinite(ratio) or ratio > 10
        rec['guard_fired'] = bool(fired)
        f40[ch] = f10[ch] if fired else p
        rec['frozen_hash'] = f40[ch].frozen_hash()
        rec['best_train_symbol_lag'] = int(max(f40[ch].symbol_alignment_diagnostic, key=f40[ch].symbol_alignment_diagnostic.get))
        info['F40g'][ch] = rec
    return {'F10k': f10, 'F40g': f40}, info


# ---------------------------------------------------------------- per-capture worker
def allowed_for(mode):
    return {'prepare': DEV, 'smoke': DEV, 'develop': DEV, 'confirm': CONF}[mode]


def run_capture(index, mode, selected=None):
    t0 = time.perf_counter()
    assert index not in RESERVE
    guard = p7.Guard(allowed_for(mode))
    limit = {'smoke': 10000, 'develop': 20000, 'confirm': None}[mode]
    rx, tx, y, labels = p7.load_raw(index, guard, limit)
    splits = {'smoke': {'train': (0, 6000), 'eval': (6000, 10000)},
              'develop': {'train': (0, 10000), 'eval': (10000, 20000)},
              'confirm': {'train': (0, 10000), 'eval': (20000, 51200)}}[mode]
    ntr = splits['train'][1]
    y_tr = y[:ntr]
    r = {'index': index, 'excluded': None, 'cells': {}, 'refs': {}, 'opened_captures': None}
    if mode == 'smoke':   # plumbing only: shrink to the training segment
        rx = {ch: v[:4 * 10000] for ch, v in rx.items()}
        c = p6.config_for(index)
        fronts = {}
        for F, n in (('F10k', 10000), ('F40g', 24000)):
            c2 = dict(c, frontend_train_samples=n)
            fronts[F] = {ch: ChannelPipeline(ch, c2).fit(rx[ch][:24000], tx[:24000], y_tr) for ch in CH}
        info = {}
    else:
        fronts, info = fit_levels(index, rx, tx, y_tr)
    r['frontends'] = info
    if fronts['F10k'] is None:
        r['excluded'] = 'F10k front-end raised; no Phase 6 baseline'
        r['opened_captures'] = guard.opened
        return r
    if mode != 'smoke':
        lags = [info['F10k'][ch]['best_train_symbol_lag'] for ch in CH]
        if any(l != 0 for l in lags):
            r['excluded'] = f'best training symbol lag {lags} != 0'
            r['opened_captures'] = guard.opened
            return r
    wanted = [(B, N) for B in ('hard', 'posterior_mean') for N in (False, True)]
    losses = {}
    for F in LEVELS:
        seg = p7.segments(fronts[F], rx, y, labels, splits)
        tr, ev = seg['train'], seg['eval']
        pred = p7.predict_family(tr, ev, wanted)
        for B, N in wanted:
            name = f'{F}_{B}_{"poly" if N else "nopoly"}'
            full = mode != 'confirm' or name in (PHASE6, (selected or {}).get('name'))
            r['cells'][name], l = p7.score_xy(pred[(B, N)], ev, per_channel=full)
            losses[name] = l['xy']
        r['refs'][f'ridge_{F}'] = p7.score_xy(pred['ridge'], ev)[0]
        if F == 'F10k':
            lms = {ch: ev['lms'][ch] for ch in CH}
            lms['xy'] = (lms['rx1'] + lms['rx2']) / 2
            r['refs']['lms_F10k'] = p7.score_xy(lms, ev)[0]
        if mode == 'confirm' and selected and selected['F'] == F:
            pc = p7.predict_family(tr, ev, [(selected['B'], selected['N'])], causal=True)
            r['refs']['selected_causal_K8_past'] = p7.score_xy(pc[(selected['B'], selected['N'])], ev)[0]
        d = dfe.hard(pred['ridge']['xy'])[ev['mask']]
        t = ev['y'][ev['mask']]
        r.setdefault('first_pass_decision_error_rate', {})[F] = float(np.mean(np.r_[d.real != t.real, d.imag != t.imag]))
    if mode == 'confirm' and selected:
        r['bootstrap_selected_vs_phase6'] = paired_blocks((losses[selected['name']] - losses[PHASE6])[:, None], 512, 2000, 404)
    r['opened_captures'] = guard.opened
    r['seconds'] = time.perf_counter() - t0
    return r


def pool_map(indices, mode, selected=None, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {i: ex.submit(run_capture, i, mode, selected) for i in indices}
        res = {}
        for i, f in futs.items():
            res[str(i)] = f.result()
            g = res[str(i)].get('frontends', {}).get('F40g', {})
            fired = [ch for ch, v in g.items() if v.get('guard_fired')]
            print(i, 'done', f"{res[str(i)].get('seconds', 0):.0f}s", res[str(i)]['excluded'] or '', 'guard fired:' if fired else '', fired or '', flush=True)
    return res


# ---------------------------------------------------------------- modes
def check_protocol():
    fz = json.loads((OUT / 'protocol_freeze.json').read_text())
    assert p6.sha(OUT / 'PROTOCOL.md') == fz['PROTOCOL.md'] and p6.sha(OUT / 'protocol.json') == fz['protocol.json']
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['status'] == 'frozen' and [c['name'] for c in p['candidates']] == [c['name'] for c in CANDS]
    return p


def write(name, data):
    p6.write_json(OUT / name, p7.clean(data))


def prepare():
    check_protocol()
    rows = {}
    for index in DEV:
        g = p7.Guard(DEV)
        rx, tx, y, _ = p7.load_raw(index, g, 10000)
        fronts, info = fit_levels(index, rx, tx, y[:10000])
        saved = {ch: ChannelPipeline.load(p6.frontend_path(index, ch), ch, p6.config_for(index)).frozen_hash() for ch in CH}
        ok = all(info['F10k'][ch]['frozen_hash'] == saved[ch] for ch in CH)
        rows[str(index)] = {'F10k_refit_matches_saved': ok,
                            'F40g': {ch: {k: info['F40g'][ch][k] for k in ('ratio', 'guard_fired')} for ch in CH}}
        print(index, 'F10k refit ok:', ok, '| guard ratio', {ch: round(info['F40g'][ch]['ratio'], 2) if info['F40g'][ch]['ratio'] else None for ch in CH},
              '| fired', [ch for ch in CH if info['F40g'][ch]['guard_fired']], flush=True)
        if not ok:
            write('prepare.json', rows)
            raise SystemExit(f'Protocol stop: F10k refit mismatch at {index}')
    write('prepare.json', rows)
    write('environment.json', p6.environment())


def smoke():
    check_protocol()
    r = run_capture(509, 'smoke')
    write('smoke.json', r)
    for k, v in r['cells'].items():
        print(f"{k:28s} XY viterbi {v['xy']['viterbi_ber']:.5f}")


def develop():
    check_protocol()
    assert not (OUT / 'develop_results.json').exists()
    res = pool_map(DEV, 'develop')
    assert all(set(r['opened_captures']) <= set(DEV) for r in res.values())
    table, logr = [], {}
    for c in CANDS:
        lr = {i: float(np.log(r['cells'][c['name']]['xy']['viterbi_ber'] / r['cells'][PHASE6]['xy']['viterbi_ber']))
              for i, r in res.items() if not r['excluded']}
        logr[c['name']] = lr
        vals = np.array(list(lr.values()))
        fresh = np.array([lr[str(i)] for i in DEV_FRESH_VAL if str(i) in lr])
        table.append({'name': c['name'], 'on': c['on'], 'F': c['F'], 'B': c['B'], 'N': c['N'],
                      'median_log_ratio': float(np.median(vals)), 'median_rel_change': float(np.expm1(np.median(vals))),
                      'wins': int(np.sum(vals < 0)), 'n': len(vals),
                      'fresh_median_rel_change': float(np.expm1(np.median(fresh))),
                      'worst_rel_change': float(np.expm1(vals.max()))})
    order = sorted(table, key=lambda t: (t['median_log_ratio'], t['on'], (t['F'] == 'F40g') * 4 + (t['B'] != 'hard') * 2 + int(t['N'])))
    top = order[0]
    sel = next(c for c in CANDS if c['name'] == top['name'])
    gate = {'a_median_rel_change': top['median_rel_change'], 'a_pass': top['median_rel_change'] <= -0.05,
            'b_wins': top['wins'], 'b_pass': top['wins'] >= 7,
            'c_fresh_median_rel_change': top['fresh_median_rel_change'], 'c_pass': top['fresh_median_rel_change'] < 0,
            'd_worst_rel_change': top['worst_rel_change'], 'd_pass': top['worst_rel_change'] <= 0.20}
    gate['pass'] = all(gate[k] for k in ('a_pass', 'b_pass', 'c_pass', 'd_pass')) and sel['name'] != PHASE6
    write('develop_results.json', res)
    write('selection.json', {'ranking': order, 'selected': sel, 'log_ratios': logr})
    write('decision.json', {'gate': gate, 'proceed_to_confirmation': gate['pass'], 'confirmation_evaluated': False})
    for t in order:
        print(f"{t['name']:28s} median {100 * t['median_rel_change']:+6.1f}%  wins {t['wins']}/{t['n']}  fresh {100 * t['fresh_median_rel_change']:+6.1f}%  worst {100 * t['worst_rel_change']:+7.1f}%")
    print('selected', sel['name'], 'gate', gate)


def code_hashes():
    files = [HERE / 'p7b.py', HERE / 'test_p7b.py', RC / 'phase7/p7.py', RC / 'decision_feedback/dfe.py',
             RC / 'decision_feedback/experiment.py'] + sorted((RC / 'src/dsp_rc').glob('*.py'))
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


def confirm():
    check_protocol()
    fz = json.loads((OUT / 'frozen.json').read_text())
    assert fz['code_sha256'] == code_hashes(), 'code changed after freeze'
    assert not (OUT / 'confirm_results.json').exists()
    sel = fz['selected']
    res = pool_map(CONF, 'confirm', sel)
    v = lambda r, n: r['cells'][n]['xy']['viterbi_ber']
    rows, counted = [], []
    for i in map(str, CONF):
        r = res[i]
        row = {'index': int(i), 'excluded': r['excluded'],
               'guard_fired': [ch for ch, x in r.get('frontends', {}).get('F40g', {}).items() if x.get('guard_fired')]}
        if not r['excluded']:
            n = r['cells'][PHASE6]['xy']['symbols']
            row.update({'phase6': v(r, PHASE6), 'selected': v(r, sel['name']),
                        'rel_change': v(r, sel['name']) / v(r, PHASE6) - 1,
                        'ridge_F10k_sbs_lower': r['refs']['ridge_F10k']['xy']['ber_lower'],
                        'ridge_F10k_viterbi': r['refs']['ridge_F10k']['xy']['viterbi_ber'],
                        'ridge_F40g_viterbi': r['refs']['ridge_F40g']['xy']['viterbi_ber'],
                        'lms_sbs_lower': r['refs']['lms_F10k']['xy']['ber_lower'],
                        'selected_causal': r['refs'].get('selected_causal_K8_past', {}).get('xy', {}).get('viterbi_ber'),
                        'phase6_bit_errors': int(round(v(r, PHASE6) * 4 * n)),
                        'ci512': r['bootstrap_selected_vs_phase6']['CI95'][0],
                        'first_pass_error_F10k': r['first_pass_decision_error_rate']['F10k']})
            if row['phase6_bit_errors'] >= 30:
                counted.append(r)
        rows.append(row)

    def contrast(a, b, rs, ref=False):
        ch = np.array([(r['refs'][a]['xy']['viterbi_ber'] / r['refs'][b]['xy']['viterbi_ber'] - 1) if ref else (v(r, a) / v(r, b) - 1) for r in rs])
        w = int(np.sum(ch < 0))
        return {'wins': w, 'n': len(ch), 'median_rel_change': float(np.median(ch)),
                'sign_test_p': float(binomtest(w, len(ch), 0.5).pvalue), 'n_worse_than_plus20pct': int(np.sum(ch > 0.20)),
                'changes': ch.tolist()}

    h2 = contrast(sel['name'], PHASE6, counted)
    h2['supported'] = (h2['wins'] / h2['n'] >= 0.8 and h2['median_rel_change'] <= -0.10 and h2['sign_test_p'] < 0.01
                       and h2['n_worse_than_plus20pct'] <= 1)
    h1 = contrast('ridge_F40g', 'ridge_F10k', counted, ref=True)
    h1['supported'] = h1['wins'] / h1['n'] >= 0.8 and h1['median_rel_change'] <= -0.05

    def factor(kind):
        pairs = {}
        for F in LEVELS:
            for o in (['nopoly', 'poly'] if kind == 'B' else ['hard', 'posterior_mean']):
                a, b = ((f'{F}_posterior_mean_{o}', f'{F}_hard_{o}') if kind == 'B' else (f'{F}_{o}_poly', f'{F}_{o}_nopoly'))
                pairs[f'{a} vs {b}'] = contrast(a, b, counted)
        pooled = float(np.median(np.concatenate([p['changes'] for p in pairs.values()])))
        ok = sum(p['median_rel_change'] < 0 and p['sign_test_p'] < 0.05 for p in pairs.values())
        return {'pairs': pairs, 'pooled_median_rel_change': pooled, 'pairs_supported': ok, 'supported': pooled < 0 and ok >= 3}

    h3, h4 = factor('B'), factor('N')
    write('confirm_results.json', {'utc': p6.now(), 'selected': sel, 'per_capture': res, 'rows': rows,
                                   'counted_captures': [r['index'] for r in counted], 'H1': h1, 'H2': h2, 'H3': h3, 'H4': h4})
    for row in rows:
        print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in row.items() if k != 'ci512'})
    for name, h in (('H1', h1), ('H2', h2)):
        print(name, {k: h[k] for k in ('wins', 'n', 'median_rel_change', 'sign_test_p', 'n_worse_than_plus20pct', 'supported')})
    for name, h in (('H3', h3), ('H4', h4)):
        print(name, round(h['pooled_median_rel_change'], 4), h['pairs_supported'], h['supported'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'smoke', 'develop', 'freeze', 'confirm'])
    {'prepare': prepare, 'smoke': smoke, 'develop': develop, 'freeze': freeze, 'confirm': confirm}[ap.parse_args().mode]()


if __name__ == '__main__':
    main()
