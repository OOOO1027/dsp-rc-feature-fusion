"""冻结二次读出双RC后追加残差RC；比较原融合、CNN融合和线性残差。"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import multiscale_runtime as public
import rx_multiscale_quadratic_rc_lms as dual
import rx_neural_dual_rc_lms as neural
from multiscale_quadratic_rc import build_features, fusion_layer
from neural_dual_rc import predict_wave

OUT = public.NN / 'dual_rc_residual_comparison'
TRAIN = (41088, 54016)
VALID = (54144, 60928)
TEST = ((15488, 51072),)
ALPHA = .001
PARAMS = dict(taps=13, passthrough_taps=25, nodes=40, gamma=.5,
              spectral_radius=.5, activation='tanh', seed=100)
BASES = ('fused', 'cnn')
MODES = ('fused', 'fused_direct_residual', 'fused_rc_residual',
         'cnn', 'cnn_direct_residual', 'cnn_rc_residual')


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def fingerprint():
    files = [neural.LOCK, neural.OUT / 'CH1_cnn.pt', neural.OUT / 'CH3_cnn.pt']
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def fit_readout(features, reference, source, columns, normalized_target, target_scale):
    margin = max(PARAMS['taps'], PARAMS['passthrough_taps']) // 2
    fit = slice(TRAIN[0] + margin, TRAIN[1] - margin)
    val = slice(VALID[0] + margin, VALID[1] - margin)
    x = features[fit][:, columns]
    y = normalized_target[fit]
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-8)
    center = float(y.mean())
    standardized = (x - mean) / std
    gram = standardized.T @ standardized / len(x)
    right = standardized.T @ (y - center) / len(x)
    beta = np.linalg.solve(gram + ALPHA * np.eye(len(columns)), right)
    raw = beta / std
    bias = center - mean @ raw
    correction = (features[:, columns] @ raw + bias) * target_scale
    output = source + correction
    if not np.isfinite(output).all():
        raise FloatingPointError('残差修正结果包含非有限数值')
    residual = reference - source[:len(reference)]
    return output, dict(coefficient=raw, bias=float(bias),
                        fit_mse=float(np.mean((residual[fit] - correction[fit]) ** 2)),
                        val_mse=float(np.mean((residual[val] - correction[val]) ** 2)),
                        val_base_mse=float(np.mean(residual[val] ** 2)),
                        train_rows=fit.stop - fit.start)


def add_residual(source, reference, case, channel, base):
    if len(reference) < VALID[1]:
        raise ValueError('参考必须覆盖训练/验证前缀')
    reference = reference[:VALID[1]]
    residual = reference - source[:len(reference)]
    input_rms = float(np.sqrt(np.mean(source[TRAIN[0]:TRAIN[1]] ** 2)))
    target_rms = float(np.sqrt(np.mean(residual[TRAIN[0]:TRAIN[1]] ** 2)))
    if min(input_rms, target_rms) < 1e-12:
        raise ValueError('残差或输入幅度过小')
    normalization = {'input_rms': input_rms,
                     'input_window_scale': max(float(np.max(abs(source[TRAIN[0]:TRAIN[1]] / input_rms))), 1e-8),
                     'target_rms': target_rms}
    features, _, _, _ = build_features(source, None, PARAMS, TRAIN[1], normalization)
    # 最后一列是偏置，读出内部独立拟合不惩罚截距。
    rc_columns = np.arange(features.shape[1] - 1)
    direct_columns = np.arange(PARAMS['nodes'], features.shape[1] - 1)
    direct, direct_info = fit_readout(features, reference, source,
                                      direct_columns, residual / target_rms, target_rms)
    rc, rc_info = fit_readout(features, reference, source,
                              rc_columns, residual / target_rms, target_rms)
    model = OUT / f'case{case:02d}_CH{1 if channel == 0 else 3}_{base}_residual.npz'
    np.savez_compressed(model, rc_coefficient=rc_info.pop('coefficient'),
                        direct_coefficient=direct_info.pop('coefficient'),
                        rc_bias=rc_info['bias'], direct_bias=direct_info['bias'],
                        params_json=json.dumps(PARAMS), normalization_json=json.dumps(normalization),
                        train_interval=np.array(TRAIN), valid_interval=np.array(VALID))
    return direct, rc, {'direct': direct_info, 'rc': rc_info, 'model': str(model)}


def evaluate_case(case, locked, cnn_networks, fp):
    record_file = OUT / f'case{case:02d}_result.json'
    if record_file.exists():
        prior = json.loads(record_file.read_text(encoding='utf-8'))
        if prior['fingerprint'] == fp:
            return prior
    waves = {name: [] for name in MODES}
    logs = {}
    for channel in range(2):
        short, long = neural.full_pair(case, channel, locked)
        selected_mode = locked['projections'][str(channel)]
        coefficients = dual.get_projection(selected_mode,
            locked['per_case_trained_projection'][str(channel)][str(case)])
        frozen_fused = fusion_layer(short, long, coefficients)
        model, amplitude = cnn_networks[channel]
        cnn_fused = predict_wave(model, short, long, scale=amplitude)
        for base, source in (('fused', frozen_fused), ('cnn', cnn_fused)):
            direct, rc, info = add_residual(source, public.COMMON.WAVEFORM[:VALID[1]],
                                            case, channel, base)
            logs[f'CH{1 if channel == 0 else 3}_{base}'] = info
            for suffix, wave in (('', source), ('_direct_residual', direct),
                                 ('_rc_residual', rc)):
                name = base + suffix
                symbols = public.backend(wave, channel)
                if len(symbols) != 51200:
                    raise ValueError('CAP / LMS 符号数不匹配')
                waves[name].append(symbols)
    results = {}
    for name, pair in waves.items():
        channel = [float(public.rates(public.channel_counts(s, public.COMMON.REFERENCES, TEST), TEST)[0])
                   for s in pair]
        pbs_fixed = float(public.rates(public.weighted_counts(*pair, np.array([.59]), TEST,
                                                                public.COMMON.REFERENCES), TEST)[0, 0])
        pbs_equal = float(public.rates(public.weighted_counts(*pair, np.array([.5]), TEST,
                                                                public.COMMON.REFERENCES), TEST)[0, 0])
        results[name] = {'ber': channel + [pbs_fixed], 'pbs_equal_weight': pbs_equal}
    record = {'case': case, 'fingerprint': fp, 'results': results, 'training': logs,
              'test_used_for_fit': False}
    save_json(record_file, record)
    print(f'case {case}/19 fused {results["fused"]["ber"][2]:.8g} → +RC {results["fused_rc_residual"]["ber"][2]:.8g}; '
          f'CNN {results["cnn"]["ber"][2]:.8g} → +RC {results["cnn_rc_residual"]["ber"][2]:.8g}', flush=True)
    return record


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    public.COMMON.init_worker()
    locked = neural.selected()
    fp = fingerprint()
    manifest_file = OUT / 'manifest.json'
    manifest = {'fingerprint': fp, 'residual_rc': PARAMS, 'ridge_alpha': ALPHA,
                'fit_wave': TRAIN, 'val_wave': VALID, 'test_symbols': TEST[0],
                'bases': BASES, 'modes': MODES, 'pbs_fixed': [.59, .41],
                'pbs_control': [.5, .5], 'correction_gain': 1.0,
                'selection': '模型和残差增益固定，未以测试 BER 调整'}
    if manifest_file.exists() and json.loads(manifest_file.read_text(encoding='utf-8')) != manifest:
        raise ValueError('已有实验目录的输入或训练配置不匹配')
    if not manifest_file.exists():
        save_json(manifest_file, manifest)
    lock_hash = hashlib.sha256(neural.LOCK.read_bytes()).hexdigest()
    cnn_networks = {ch: neural.load_network(ch, 'cnn', lock_hash) for ch in range(2)}
    matrices = {name: np.empty((19, 3)) for name in MODES}
    equal = {name: np.empty(19) for name in MODES}
    training = []
    for case in dual.CASES:
        result = evaluate_case(case, locked, cnn_networks, fp)
        training.append(result['training'])
        for name in MODES:
            matrices[name][case - 1] = result['results'][name]['ber']
            equal[name][case - 1] = result['results'][name]['pbs_equal_weight']
    expected = {'fused': public.NN / 'multiscale_quadratic_fused_test_berall.txt',
                'cnn': neural.OUT / 'cnn_test_berall.txt'}
    for name, path in expected.items():
        np.testing.assert_array_equal(matrices[name], np.loadtxt(path))
    for name in MODES:
        if not np.isfinite(matrices[name]).all():
            raise ValueError(f'{name} 不完整')
        np.savetxt(OUT / f'{name}_test_berall.txt', matrices[name], fmt='%.16e')
        np.savetxt(OUT / f'{name}_equal_pbs_test_berall.txt', equal[name], fmt='%.16e')
    summary = {'means': {name: matrices[name].mean(axis=0).tolist() for name in MODES},
               'equal_pbs_means': {name: float(equal[name].mean()) for name in MODES},
               'paired_pbs': {}, 'training': training,
               'baseline_exact_replay_all_19': True,
               'comparison_limits': '残差级增加模型容量和拟合预算；当前验证/测试数据多轮复用，结果属于探索性回放'}
    for base in BASES:
        for kind in ('direct', 'rc'):
            name = f'{base}_{kind}_residual'
            summary['paired_pbs'][name + '_minus_' + base] = public.paired_statistics(
                matrices[name][:, 2] - matrices[base][:, 2])
    save_json(OUT / 'summary.json', summary)
    print(json.dumps({key: {'CH1': val[0], 'CH3': val[1], 'PBS': val[2]}
                      for key, val in summary['means'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
