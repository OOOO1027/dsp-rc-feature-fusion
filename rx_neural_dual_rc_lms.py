"""在冻结的短/长 RC 波形上训练 CNN 与 Bi-LSTM，并沿用原 CAP/LMS/PBS 严格评价。"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import multiscale_runtime as public
import rx_multiscale_quadratic_rc_lms as existing
from neural_dual_rc import WINDOW, HALO, CORE, make_model, predict_wave

OUT = public.NN / 'neural_dual_rc'
LOCK = public.NN / 'multiscale_quadratic_selection.json'
TRAIN = (41088, 54016)
VALID = (54144, 60928)
TEST = ((15488, 51072),)
SEED = 20261025
EPOCHS = 12
PATIENCE = 3
NAMES = ('cnn', 'bilstm')


def write_json(path, item):
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding='utf-8')


def selected():
    item = json.loads(LOCK.read_text(encoding='utf-8'))
    if item['pbs_CH1_weight'] != .59 or item['test_slice'] != [15488, 51072]:
        raise ValueError('冻结 RC / PBS 口径发生变化')
    return item


def scale():
    return float(np.sqrt(np.mean(public.COMMON.WAVEFORM[:40960] ** 2)))


def prefix_pair(case, channel, lock):
    waves = []
    for branch in existing.BRANCHES:
        path = existing.wave_path(case, channel, branch)
        with np.load(path, allow_pickle=False) as data:
            alpha = lock['alphas'][str(channel)][branch]
            matches = np.flatnonzero(np.isclose(data['alphas'], alpha, atol=0, rtol=1e-12))
            if len(matches) != 1:
                raise ValueError(f'{path} 缺少锁定岭强度 {alpha}')
            waves.append(data['waves'][:, matches[0]].astype(np.float32))
    if any(len(w) != 61440 for w in waves):
        raise ValueError('分支训练前缀长度错误')
    return np.column_stack(waves)


def windows(pair, target, bounds, amplitude):
    starts = range(bounds[0], bounds[1] - WINDOW + 1, CORE)
    inputs = np.stack([pair[s:s + WINDOW] for s in starts]).astype(np.float32) / amplitude
    labels = np.stack([target[s + HALO:s + WINDOW - HALO] for s in starts]).astype(np.float32) / amplitude
    return inputs, labels


def assemble(channel, lock, amplitude):
    target = public.COMMON.WAVEFORM
    train_x, train_y, val_x, val_y = [], [], [], []
    for case in existing.CASES:
        pair = prefix_pair(case, channel, lock)
        x, y = windows(pair, target, TRAIN, amplitude)
        vx, vy = windows(pair, target, VALID, amplitude)
        train_x.append(x); train_y.append(y); val_x.append(vx); val_y.append(vy)
    return tuple(torch.from_numpy(np.concatenate(items)) for items in (train_x, train_y, val_x, val_y))


def score(model, loader):
    model.eval()
    total = 0.; count = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x)[:, HALO:WINDOW - HALO]
            total += float(torch.sum((pred - y) ** 2)); count += y.numel()
    return total / count


def train_models(lock, *, force=False):
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(SEED); np.random.seed(SEED)
    public.COMMON.init_worker()
    amplitude = scale()
    manifest = {'seed': SEED, 'train_wave': TRAIN, 'validation_wave': VALID,
                'strict_test_symbols': TEST[0], 'window': WINDOW, 'halo': HALO,
                'cases': list(existing.CASES), 'amplitude': amplitude,
                'lock_sha256': hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                'epochs_max': EPOCHS, 'patience': PATIENCE, 'optimizer': 'Adam(lr=0.002, weight_decay=1e-5)',
                'loss': 'center waveform MSE + 0.02 * squared correction',
                'selection': 'smallest validation waveform MSE; strict test never selects an epoch',
                'pytorch': torch.__version__, 'device': 'cpu'}
    old = OUT / 'manifest.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8')) != manifest:
        raise ValueError('已有神经实验目录的冻结数据/训练配置不匹配')
    if not old.exists():
        write_json(old, manifest)
    records = {}
    for channel in range(2):
        tx, ty, vx, vy = assemble(channel, lock, amplitude)
        train_loader = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(SEED + channel))
        val_loader = DataLoader(TensorDataset(vx, vy), batch_size=64, shuffle=False)
        print(f'CH{1 if channel == 0 else 3} 训练窗={len(tx)} 验证窗={len(vx)}', flush=True)
        for name in NAMES:
            file = OUT / f'CH{1 if channel == 0 else 3}_{name}.pt'
            history_file = OUT / f'CH{1 if channel == 0 else 3}_{name}_history.json'
            if file.exists() and history_file.exists() and not force:
                records[f'CH{1 if channel == 0 else 3}_{name}'] = json.loads(history_file.read_text(encoding='utf-8'))
                continue
            model = make_model(name)
            optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-5)
            best = float('inf'); stale = 0; history = []
            began = time.perf_counter()
            for epoch in range(1, EPOCHS + 1):
                model.train(); train_loss = 0.; steps = 0
                for x, y in train_loader:
                    pred = model(x)[:, HALO:WINDOW - HALO]
                    base = x[:, HALO:WINDOW - HALO, 0]
                    loss = torch.mean((pred - y) ** 2) + .02 * torch.mean((pred - base) ** 2)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step(); train_loss += float(loss.detach()); steps += 1
                validation = score(model, val_loader)
                history.append({'epoch': epoch, 'train_objective': train_loss / steps, 'val_wave_mse': validation})
                print(f'{name} CH{1 if channel == 0 else 3} epoch {epoch}: train={train_loss / steps:.6g} val={validation:.6g}', flush=True)
                if validation < best - 1e-7:
                    best = validation; stale = 0
                    torch.save({'model': name, 'channel': channel, 'scale': amplitude,
                                'state_dict': model.state_dict(), 'epoch': epoch,
                                'lock_sha256': manifest['lock_sha256']}, file)
                else:
                    stale += 1
                    if stale >= PATIENCE:
                        break
            record = {'model': name, 'channel': channel, 'best_val_wave_mse': best,
                      'best_epoch': min(history, key=lambda row: row['val_wave_mse'])['epoch'],
                      'history': history, 'seconds': time.perf_counter() - began,
                      'parameter_count': sum(p.numel() for p in model.parameters())}
            write_json(history_file, record)
            records[f'CH{1 if channel == 0 else 3}_{name}'] = record
    write_json(OUT / 'training_summary.json', records)
    return records


def load_network(channel, name, lock_hash):
    payload = torch.load(OUT / f'CH{1 if channel == 0 else 3}_{name}.pt', map_location='cpu', weights_only=True)
    if payload['lock_sha256'] != lock_hash or payload['model'] != name or payload['channel'] != channel:
        raise ValueError('模型与冻结 RC 配置不一致')
    model = make_model(name); model.load_state_dict(payload['state_dict']); model.eval()
    return model, float(payload['scale'])


def full_pair(case, channel, lock):
    file = OUT / f'case{case:02d}_CH{1 if channel == 0 else 3}_branches.npz'
    if file.exists():
        with np.load(file, allow_pickle=False) as data:
            if data['lock_sha256'].item() == hashlib.sha256(LOCK.read_bytes()).hexdigest():
                return data['short'].copy(), data['long'].copy()
    wave = public.COMMON.load(f'rxdata/rx{channel + 1}_{case}.txt')
    branches = [existing.frozen_branch(case, channel, name, lock['alphas'][str(channel)][name], wave)
                for name in existing.BRANCHES]
    if any(len(x) != 204800 or not np.isfinite(x).all() for x in branches):
        raise ValueError('冻结分支的完整波形无效')
    np.savez_compressed(file, short=branches[0], long=branches[1],
                        lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest())
    return branches


def evaluate(lock, cases):
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    public.COMMON.init_worker()
    lock_hash = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    networks = {(channel, name): load_network(channel, name, lock_hash)
                for channel in range(2) for name in NAMES}
    rows = {name: np.full((19, 3), np.nan) for name in NAMES}
    val = {name: [] for name in NAMES}
    for case in cases:
        case_file = OUT / f'case{case:02d}_result.json'
        if case_file.exists():
            saved = json.loads(case_file.read_text(encoding='utf-8'))
            if saved['lock_sha256'] == lock_hash:
                for name in NAMES:
                    rows[name][case - 1] = saved['test'][name]
                    val[name].append(saved['confirmation'][name])
                print(f'case {case}/19 cached', flush=True)
                continue
        symbols = {name: [] for name in NAMES}
        for channel in range(2):
            short, long = full_pair(case, channel, lock)
            for name in NAMES:
                model, amplitude = networks[channel, name]
                fused = predict_wave(model, short, long, scale=amplitude)
                symbols[name].append(public.backend(fused, channel))
        saved = {'case': case, 'lock_sha256': lock_hash, 'test': {}, 'confirmation': {}}
        for name, pair in symbols.items():
            ch = [float(public.rates(public.channel_counts(s, public.COMMON.REFERENCES, TEST), TEST)[0])
                  for s in pair]
            pbs = float(public.rates(public.weighted_counts(*pair, np.array([.59]), TEST,
                                                         public.COMMON.REFERENCES), TEST)[0, 0])
            result = ch + [pbs]
            rows[name][case - 1] = result; saved['test'][name] = result
            confirm = public.CHECK
            counts = public.weighted_counts(*pair, np.array([.59]), confirm, public.COMMON.REFERENCES)
            saved['confirmation'][name] = public.rates(counts, confirm)[0].tolist()
            val[name].append(saved['confirmation'][name])
        write_json(case_file, saved)
        print(f'case {case}/19 CNN PBS={rows["cnn"][case-1,2]:.8g} Bi-LSTM PBS={rows["bilstm"][case-1,2]:.8g}', flush=True)
    if len(cases) == 19:
        short = np.loadtxt(public.NN / 'multiscale_quadratic_short_only_test_berall.txt')
        prior = np.loadtxt(public.NN / 'multiscale_quadratic_fused_test_berall.txt')
        summary = {'protocol': 'same frozen RC branches / CAP / symbol LMS / PBS=.59 / strict test',
                   'test_slice': TEST[0], 'training_wave': TRAIN, 'validation_wave': VALID,
                   'means': {}, 'paired_vs_short': {}, 'short_only_mean': short.mean(axis=0).tolist(),
                   'prior_locked_fusion_mean': prior.mean(axis=0).tolist(),
                   'prefix_PBS_segment_means': {},
                   'prefix_segment_note': '第一段 [12800,14016) 与神经训练波形部分重叠，不作独立验证；第二段 [14016,15232) 在神经留出区内'}
        for name, array in rows.items():
            if array.shape != (19, 3) or not np.isfinite(array).all():
                raise ValueError('神经方案测试矩阵不完整')
            np.savetxt(OUT / f'{name}_test_berall.txt', array, fmt='%.16e')
            summary['means'][name] = array.mean(axis=0).tolist()
            summary['paired_vs_short'][name] = public.paired_statistics(array[:, 2] - short[:, 2])
            summary['prefix_PBS_segment_means'][name] = np.asarray(val[name]).mean(axis=0).tolist()
        write_json(OUT / 'result_summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--cases', nargs='*', type=int, default=list(existing.CASES))
    args = parser.parse_args()
    if args.train_only and args.evaluate_only:
        parser.error('训练与仅评价参数互斥')
    if sorted(set(args.cases)) != sorted(args.cases) or any(x not in existing.CASES for x in args.cases):
        parser.error('实验组须为 1–19 内不重复编号')
    lock = selected()
    if not args.evaluate_only:
        train_models(lock)
    if not args.train_only:
        evaluate(lock, args.cases)


if __name__ == '__main__':
    main()
