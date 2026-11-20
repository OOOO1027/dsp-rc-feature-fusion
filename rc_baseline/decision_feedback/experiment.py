"""Phase 6: decision-feedback two-branch equalizer + NLTCP trellis decoding.

Modes (see runs/phase6_decision_feedback/PROTOCOL.md):
  prepare   hash inputs/front-ends for all 10 captures (no label parsing)
  smoke     509 training segment only: fit [0,6000), score [6000,10000)
  validate  509 train/validation; selection, gate, H3; never parses test labels
  freeze    write frozen.json (only if the validation gate passed)
  test      one-shot evaluation of the frozen design on 10 historical test segments
"""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RC = HERE.parent
ROOT = RC.parent
sys.path.insert(0, str(RC / 'src'))
sys.path.insert(0, str(HERE))

from dsp_rc.experiment import ChannelPipeline, mask_for          # noqa: E402
from dsp_rc.fusion import FusionModel, candidates as phase3_candidates  # noqa: E402
from dsp_rc.phase4 import paired_blocks                          # noqa: E402
import dfe                                                       # noqa: E402

OUT = RC / 'runs/phase6_decision_feedback'
VPP = Path.home() / 'Documents/ChatGPT/DSP/deliverables/DSP_Vpp九点与跨偏振_20261113/inputs'
NROOT = ROOT / 'NN数据20250320/NN数据20250320'
CH = ('rx1', 'rx2')
GUARD = 128
TEST_INDICES = [509, 513, 517, 519, 523, 528, 529, 533, 535, 382]
REFERENCES = ('lms', 'ridge_low', 'linear80_low', 'dual40_reg')


# ---------------------------------------------------------------- io helpers
def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def write_json(path, data, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(path.read_text())


def first_tokens(path, n):
    """Parse only the first n whitespace-separated numbers of a text file."""
    out, tail = [], ''
    with open(path, 'r') as f:
        while len(out) < n:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            parts = (tail + chunk).split()
            if chunk[-1].isspace():
                tail = ''
            else:
                tail = parts.pop() if parts else ''
            out.extend(parts[:n - len(out)])
        if len(out) < n and tail:
            out.append(tail)
    if len(out) < n:
        raise ValueError(f'{path} has fewer than {n} tokens')
    return np.array(out[:n], dtype=float)


def config_for(index):
    c = read_json(RC / 'runs/index509_fixed_v1/config.json')
    c['index'] = index
    return c


def frontend_path(index, ch):
    if index == 509:
        return RC / f'runs/index509_fixed_v1/model_{ch}.npz'
    if index in (513, 382):
        return RC / f'runs/phase3_feature_fusion/index{index}/frontend_{ch}.npz'
    return VPP / f'index{index}/frontend_{ch}.npz'


def raw_paths(index):
    return {'rx1': NROOT / f'rxdata/rx1_PBS_NLTCP{index}.txt', 'rx2': NROOT / f'rxdata/rx2_PBS_NLTCP{index}.txt',
            'i': NROOT / 'v1.txt', 'q': NROOT / 'v2.txt',
            'pam_i': NROOT / 'decimal_PAM4_1.txt', 'pam_q': NROOT / 'decimal_PAM4_2.txt'}


def audit_hashes():
    rows = csv.DictReader((ROOT / 'audit_outputs/2026-10-16/file_index.csv').open())
    return {r['path']: r['sha256'] for r in rows}


class LabelAccess:
    """Counts how many label tokens beyond the development range are parsed."""
    def __init__(self):
        self.beyond_20000 = 0


def load_condition(index, splits, label_limit, access, manifest):
    """Returns CAP symbols, LMS outputs, ideal symbols and labels per split."""
    audit = audit_hashes()
    paths = raw_paths(index)
    for k, p in paths.items():
        h = sha(p)
        if audit.get(str(p)) != h:
            raise RuntimeError(f'Raw file hash mismatch: {p}')
    c = config_for(index)
    rx = {ch: np.loadtxt(paths[ch]).ravel() for ch in CH}          # inputs only
    assert all(len(v) == 204800 for v in rx.values())
    if label_limit is None:
        lab = {k: np.loadtxt(paths[k]).ravel() for k in ['i', 'q', 'pam_i', 'pam_q']}
        access.beyond_20000 += 4 * (51200 - 20000)
    else:
        assert label_limit <= 20000
        lab = {k: first_tokens(paths[k], label_limit) for k in ['i', 'q', 'pam_i', 'pam_q']}
    y = lab['i'] + 1j * lab['q']
    labels = np.column_stack([lab['pam_i'], lab['pam_q']]).astype(int)
    fronts, entry = {}, {'raw': {k: str(p) for k, p in paths.items()}, 'frontends': {}}
    for ch in CH:
        fp = frontend_path(index, ch)
        fronts[ch] = ChannelPipeline.load(fp, ch, c)
        entry['frontends'][ch] = {'path': str(fp), 'sha256': sha(fp), 'frozen_hash': fronts[ch].frozen_hash(),
                                  'alignment': fronts[ch].alignment}
    manifest[str(index)] = entry
    data = {}
    for split, (a, b) in splits.items():
        d = {'z': {}, 'lms': {}, 'range': [a, b]}
        for ch in CH:
            z = fronts[ch].preprocess(rx[ch][4 * a:4 * b], a)[1]
            d['z'][ch] = z
            d['lms'][ch] = fronts[ch].lms.predict(z) * fronts[ch].target_scale
        d['y'] = y[a:b]
        d['labels'] = labels[a:b]
        d['mask'] = mask_for(b - a, GUARD)
        data[split] = d
    return data


# ---------------------------------------------------------------- modelling
def ref_config(name):
    return next(c.copy() for c in phase3_candidates() if c['name'] == name)


def build_predictions(tr, ev, cands, refs=REFERENCES):
    """Fit everything on `tr`, predict on `ev`. Returns {name: {rx1,rx2,xy}} and diagnostics."""
    pred, diag = {}, {'first_pass_decision_error_rate': {}, 'fit_seconds': {}, 'predict_seconds': {}}
    p0_cf, p0_ev, U = {}, {}, {}
    for ch in CH:
        win = dfe.Window(tr['z'][ch], tr['mask'])
        U[ch] = (win(tr['z'][ch]), win(ev['z'][ch]))
        W0 = dfe.ridge_fit(dfe.with_bias(U[ch][0]), tr['y'], tr['mask'])
        p0_ev[ch] = dfe.ridge_apply(dfe.with_bias(U[ch][1]), W0)
        p0_cf[ch] = dfe.crossfit_first_pass(U[ch][0], tr['y'], tr['mask'])
    p0_ev['xy'] = (p0_ev['rx1'] + p0_ev['rx2']) / 2
    p0_cf['xy'] = (p0_cf['rx1'] + p0_cf['rx2']) / 2
    for k in ['rx1', 'rx2', 'xy']:
        d = dfe.hard(p0_ev[k])[ev['mask']]
        t = ev['y'][ev['mask']]
        diag['first_pass_decision_error_rate'][k] = float(np.mean(np.r_[d.real != t.real, d.imag != t.imag]))

    def put(name, per_ch):
        per_ch['xy'] = (per_ch['rx1'] + per_ch['rx2']) / 2
        pred[name] = per_ch

    for name in refs:
        t = time.perf_counter()
        if name == 'lms':
            put(name, {ch: ev['lms'][ch] for ch in CH})
        else:
            out = {}
            for ch in CH:
                m = FusionModel(ref_config(name)).fit(tr['z'][ch], tr['y'], tr['mask'])
                out[ch] = m.predict(ev['z'][ch])
            put(name, out)
            if name == 'ridge_low':   # sanity: identical to the first-pass ridge
                for ch in CH:
                    np.testing.assert_allclose(out[ch], p0_ev[ch], rtol=0, atol=1e-8)
        diag['fit_seconds'][name] = time.perf_counter() - t

    for c in cands:
        seeds = c.get('seeds', [None])
        for seed in seeds:
            key = c['name'] if seed is None else f"{c['name']}_s{seed}"
            t = time.perf_counter()
            out = {}
            for ch in CH:
                src = 'xy' if c['feedback_from'] == 'xy_first_pass' else ch
                dec_ev = dfe.hard(p0_ev[src])
                dec_tr = tr['y'] if c['train_feedback'] == 'teacher' else dfe.hard(p0_cf[src])
                if c['family'] == 'DF-RC':
                    B_tr = dfe.feedback_reservoir(dec_tr, c['activation'], seed, c['nodes'])
                    B_ev = dfe.feedback_reservoir(dec_ev, c['activation'], seed, c['nodes'])
                else:
                    B_tr = dfe.feedback_taps(dec_tr, c['K'], c['span'])
                    B_ev = dfe.feedback_taps(dec_ev, c['K'], c['span'])
                W = dfe.ridge_fit(dfe.with_bias(U[ch][0], B_tr), tr['y'], tr['mask'], c['ridge'])
                out[ch] = dfe.ridge_apply(dfe.with_bias(U[ch][1], B_ev), W)
            put(key, out)
            diag['fit_seconds'][key] = time.perf_counter() - t
    return pred, diag


def score_all(pred, ev):
    metrics, losses = {}, {}
    for name, per in pred.items():
        metrics[name], losses[name] = {}, {}
        for k, z in per.items():
            metrics[name][k], losses[name][k] = dfe.score(z, ev['y'], ev['labels'], ev['mask'])
            metrics[name][k]['viterbi_error_runs'] = dfe.error_runs(losses[name][k]['viterbi_ber'])
    return metrics, losses


def candidate_metric(metrics, c, out, metric):
    if 'seeds' in c:
        return float(np.mean([metrics[f"{c['name']}_s{s}"][out][metric] for s in c['seeds']]))
    return metrics[c['name']][out][metric]


def candidate_losses(losses, c, out, metric):
    if 'seeds' in c:
        return np.mean([losses[f"{c['name']}_s{s}"][out][metric] for s in c['seeds']], axis=0)
    return losses[c['name']][out][metric]


def boot(delta):
    return {str(L): paired_blocks(np.asarray(delta)[:, None], L, 2000, 404) for L in (256, 512)}


def ci_below_zero(b):
    return all(b[str(L)]['CI95'][0][1] < 0 for L in (256, 512))


# ---------------------------------------------------------------- modes
def environment():
    import scipy
    return {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'utc': now()}


def check_protocol():
    freeze = read_json(OUT / 'protocol_freeze.json')
    assert sha(OUT / 'PROTOCOL.md') == freeze['PROTOCOL.md'], 'PROTOCOL.md changed after freeze'
    assert sha(OUT / 'protocol.json') == freeze['protocol.json'], 'protocol.json changed after freeze'
    p = read_json(OUT / 'protocol.json')
    assert p['status'] == 'frozen' and len(p['candidates']) == 19
    return p


def prepare():
    check_protocol()
    manifest = {}
    audit = audit_hashes()
    for index in TEST_INDICES:
        e = {'raw': {}, 'frontends': {}}
        for k, p in raw_paths(index).items():
            h = sha(p)
            assert audit.get(str(p)) == h, p
            e['raw'][k] = {'path': str(p), 'sha256': h}
        c = config_for(index)
        for ch in CH:
            fp = frontend_path(index, ch)
            e['frontends'][ch] = {'path': str(fp), 'sha256': sha(fp),
                                  'frozen_hash': ChannelPipeline.load(fp, ch, c).frozen_hash()}
        manifest[str(index)] = e
    write_json(OUT / 'input_manifest.json', manifest)
    write_json(OUT / 'environment.json', environment())
    print('prepared', len(manifest), 'captures')


def smoke():
    p = check_protocol()
    access, manifest = LabelAccess(), {}
    data = load_condition(509, {'train': (0, 6000), 'dev': (6000, 10000)}, 10000, access, manifest)
    cands = [c for c in p['candidates'] if c['name'] in ('dftap_K8_crossfit_past', 'dfrc40_identity_crossfit')]
    pred, diag = build_predictions(data['train'], data['dev'], cands, refs=('lms', 'ridge_low'))
    metrics, _ = score_all(pred, data['dev'])
    rows = {k: {o: {m: v[o][m] for m in ('mse', 'ber_lower', 'viterbi_ber')} for o in v} for k, v in metrics.items()}
    write_json(OUT / 'smoke.json', {'utc': now(), 'scope': '509 training segment only: fit [0,6000), score [6000,10000)',
                                    'label_tokens_beyond_20000': access.beyond_20000, 'metrics': rows,
                                    'first_pass_decision_error_rate': diag['first_pass_decision_error_rate']})
    for k, v in rows.items():
        print(f"{k:34s} XY MSE {v['xy']['mse']:.4f} lower {v['xy']['ber_lower']:.5f} viterbi {v['xy']['viterbi_ber']:.5f}")


def validate():
    p = check_protocol()
    assert not (OUT / 'validation_results.json').exists()
    t0 = time.perf_counter()
    access, manifest = LabelAccess(), {}
    data = load_condition(509, {'train': (0, 10000), 'validation': (10000, 20000)}, 20000, access, manifest)
    assert access.beyond_20000 == 0
    cands = p['candidates']
    pred, diag = build_predictions(data['train'], data['validation'], cands)
    metrics, losses = score_all(pred, data['validation'])

    table = []
    for c in cands:
        row = {'name': c['name'], 'family': c['family']}
        for out in ['rx1', 'rx2', 'xy']:
            for m in ['mse', 'ber_lower', 'ber_upper', 'viterbi_ber']:
                row[f'{out}_{m}'] = candidate_metric(metrics, c, out, m)
        row['feature_dim'] = 231 + (c['nodes'] if c['family'] == 'DF-RC' else 2 * c['K'] * (2 if c['span'] == 'past+future' else 1))
        row['K'] = c.get('K', 0)
        table.append(row)
    order = sorted(table, key=lambda r: (r['xy_viterbi_ber'], (r['rx1_viterbi_ber'] + r['rx2_viterbi_ber']) / 2,
                                         r['feature_dim'], r['K']))
    selected = next(c for c in cands if c['name'] == order[0]['name'])

    base = {o: metrics['ridge_low'][o]['viterbi_ber'] for o in ['rx1', 'rx2', 'xy']}
    sel = {o: candidate_metric(metrics, selected, o, 'viterbi_ber') for o in ['rx1', 'rx2', 'xy']}
    delta = candidate_losses(losses, selected, 'xy', 'viterbi_ber') - losses['ridge_low']['xy']['viterbi_ber']
    b = boot(delta)
    gate = {'a_xy_relative_reduction': 1 - sel['xy'] / base['xy'],
            'a_pass': (1 - sel['xy'] / base['xy']) >= 0.10,
            'b_pass': sel['rx1'] < base['rx1'] and sel['rx2'] < base['rx2'],
            'c_bootstrap': b, 'c_pass': ci_below_zero(b)}
    gate['pass'] = gate['a_pass'] and gate['b_pass'] and gate['c_pass']

    # H3: DF-RC vs best crossfit / past DF-tap (same source, same span).
    taps = [r for r in table if r['family'] == 'DF-tap' and '_crossfit_past' in r['name'] and not r['name'].endswith('future')]
    best_tap = min(taps, key=lambda r: r['xy_viterbi_ber'])
    h3 = {'reference_tap': best_tap['name']}
    for c in [c for c in cands if c['family'] == 'DF-RC']:
        d = candidate_losses(losses, c, 'xy', 'viterbi_ber') - losses[best_tap['name']]['xy']['viterbi_ber']
        bb = boot(d)
        h3[c['name']] = {'xy_viterbi_ber_3seed_mean': candidate_metric(metrics, c, 'xy', 'viterbi_ber'),
                         'reference_xy_viterbi_ber': best_tap['xy_viterbi_ber'], 'bootstrap': bb,
                         'supported': bool(np.mean(d) < 0 and ci_below_zero(bb))}
    h3['supported_any'] = any(v['supported'] for k, v in h3.items() if isinstance(v, dict))

    # 2x2 decomposition on validation, XY.
    decomposition = {
        'ridge_sbs_lower': metrics['ridge_low']['xy']['ber_lower'], 'ridge_sbs_upper': metrics['ridge_low']['xy']['ber_upper'],
        'ridge_viterbi': base['xy'],
        'selected_sbs_lower': candidate_metric(metrics, selected, 'xy', 'ber_lower'),
        'selected_sbs_upper': candidate_metric(metrics, selected, 'xy', 'ber_upper'),
        'selected_viterbi': sel['xy']}

    np.savez_compressed(OUT / 'validation_predictions.npz', **{f'{k}_{o}': v for k, per in pred.items() for o, v in per.items()},
                        mask=data['validation']['mask'], symbol_ids=np.arange(10000, 20000))
    write_json(OUT / 'validation_results.json', {'utc': now(), 'metrics': metrics, 'diagnostics': diag,
                                                 'label_tokens_beyond_20000': access.beyond_20000,
                                                 'inputs': manifest, 'seconds': time.perf_counter() - t0})
    write_json(OUT / 'selection.json', {'utc': now(), 'ranking': order, 'selected': selected,
                                        'references_xy': {r: metrics[r]['xy'] for r in REFERENCES},
                                        'decomposition_xy': decomposition})
    write_json(OUT / 'decision.json', {'utc': now(), 'gate': gate, 'H3_validation': h3,
                                       'proceed_to_test': gate['pass'], 'test_evaluated': False})
    print('selected', selected['name'], 'gate', gate['pass'], {k: gate[k] for k in ['a_xy_relative_reduction', 'a_pass', 'b_pass', 'c_pass']})
    print('H3', {k: (v['supported'] if isinstance(v, dict) else v) for k, v in h3.items()})


def code_hashes():
    files = [HERE / 'dfe.py', HERE / 'experiment.py', HERE / 'test_dfe.py'] + sorted((RC / 'src/dsp_rc').glob('*.py'))
    return {str(f.relative_to(ROOT)): sha(f) for f in files}


def freeze():
    check_protocol()
    d = read_json(OUT / 'decision.json')
    if not d['proceed_to_test']:
        raise SystemExit('Validation gate failed: protocol forbids test evaluation.')
    s = read_json(OUT / 'selection.json')
    write_json(OUT / 'frozen.json', {'utc': now(), 'selected': s['selected'], 'code_sha256': code_hashes(),
                                     'protocol_freeze': read_json(OUT / 'protocol_freeze.json'),
                                     'input_manifest_sha256': sha(OUT / 'input_manifest.json'),
                                     'validation_results_sha256': sha(OUT / 'validation_results.json'),
                                     'test_indices': TEST_INDICES})
    print('frozen', s['selected']['name'])


def test():
    check_protocol()
    fz = read_json(OUT / 'frozen.json')
    assert fz['code_sha256'] == code_hashes(), 'Code changed after freeze'
    assert not (OUT / 'test_results.json').exists()
    sel = fz['selected']
    access, manifest = LabelAccess(), {}
    per_index, rows = {}, []
    for index in TEST_INDICES:
        t = time.perf_counter()
        data = load_condition(index, {'train': (0, 10000), 'test': (20000, 51200)}, None, access, manifest)
        pred, diag = build_predictions(data['train'], data['test'], [sel])
        metrics, losses = score_all(pred, data['test'])
        selm = {o: {m: candidate_metric(metrics, sel, o, m) for m in ['mse', 'ber_lower', 'ber_upper', 'viterbi_ber']}
                for o in ['rx1', 'rx2', 'xy']}
        sl = candidate_losses(losses, sel, 'xy', 'viterbi_ber')
        b_h1 = boot(sl - losses['ridge_low']['xy']['ber_lower'])
        b_h2 = boot(sl - losses['ridge_low']['xy']['viterbi_ber'])
        per_index[str(index)] = {'metrics': metrics, 'selected_metrics': selm, 'diagnostics': diag,
                                 'bootstrap_vs_ridge_sbs_lower': b_h1, 'bootstrap_vs_ridge_viterbi': b_h2,
                                 'seconds': time.perf_counter() - t}
        r = {'index': index, 'ridge_sbs_lower': metrics['ridge_low']['xy']['ber_lower'],
             'ridge_sbs_upper': metrics['ridge_low']['xy']['ber_upper'],
             'ridge_viterbi': metrics['ridge_low']['xy']['viterbi_ber'],
             'lms_sbs_lower': metrics['lms']['xy']['ber_lower'],
             'dual40_reg_sbs_lower': metrics['dual40_reg']['xy']['ber_lower'],
             'selected_viterbi': selm['xy']['viterbi_ber'], 'selected_sbs_lower': selm['xy']['ber_lower'],
             'ci512_vs_ridge_viterbi': b_h2['512']['CI95'][0],
             'ridge_viterbi_errors_bits': int(round(metrics['ridge_low']['xy']['viterbi_ber'] * 4 * metrics['ridge_low']['xy']['symbols'])),
             'selected_viterbi_errors_bits': int(round(selm['xy']['viterbi_ber'] * 4 * metrics['ridge_low']['xy']['symbols']))}
        rows.append(r)
        print(index, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != 'ci512_vs_ridge_viterbi'}, flush=True)
    rel1 = [r['selected_viterbi'] / r['ridge_sbs_lower'] - 1 for r in rows]
    h1 = {'wins': sum(r['selected_viterbi'] < r['ridge_sbs_lower'] for r in rows), 'median_relative_change': float(np.median(rel1))}
    h1['supported'] = h1['wins'] >= 8 and h1['median_relative_change'] <= -0.15
    rel2 = [r['selected_viterbi'] / r['ridge_viterbi'] - 1 for r in rows]
    h2 = {'wins': sum(r['selected_viterbi'] < r['ridge_viterbi'] for r in rows),
          'ci512_below_zero': sum(r['ci512_vs_ridge_viterbi'][1] < 0 for r in rows),
          'median_relative_change': float(np.median(rel2))}
    h2['supported'] = h2['wins'] >= 8 and h2['ci512_below_zero'] >= 6
    write_json(OUT / 'test_results.json', {'utc': now(), 'selected': sel['name'], 'per_index': per_index, 'summary_rows': rows,
                                           'H1': h1, 'H2': h2, 'inputs': manifest,
                                           'label_note': 'historical test segments unused for this method; same Tx payload'})
    print('H1', h1); print('H2', h2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'smoke', 'validate', 'freeze', 'test'])
    {'prepare': prepare, 'smoke': smoke, 'validate': validate, 'freeze': freeze, 'test': test}[ap.parse_args().mode]()


if __name__ == '__main__':
    main()
