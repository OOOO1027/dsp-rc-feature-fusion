"""事后诊断：同架构 CNN 只看短路，检验长路是否提供额外收益。"""
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import rx_neural_dual_rc_lms as run
from neural_dual_rc import HALO, WINDOW, make_model, predict_wave

NAME = 'cnn_short_input'


def train(lock):
    torch.set_num_threads(4)
    torch.manual_seed(run.SEED)
    run.public.COMMON.init_worker()
    amplitude = run.scale()
    for channel in range(2):
        file = run.OUT / f'CH{1 if channel == 0 else 3}_{NAME}.pt'
        if file.exists():
            continue
        tx, ty, vx, vy = run.assemble(channel, lock, amplitude)
        tx[:, :, 1] = tx[:, :, 0]
        vx[:, :, 1] = vx[:, :, 0]
        loader = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(run.SEED + channel))
        validation = DataLoader(TensorDataset(vx, vy), batch_size=64, shuffle=False)
        model = make_model('cnn')
        optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-5)
        best = float('inf'); stale = 0; history = []
        began = time.perf_counter()
        for epoch in range(1, run.EPOCHS + 1):
            model.train(); train_loss = 0.; steps = 0
            for x, y in loader:
                pred = model(x)[:, HALO:WINDOW - HALO]
                base = x[:, HALO:WINDOW - HALO, 0]
                loss = torch.mean((pred - y) ** 2) + .02 * torch.mean((pred - base) ** 2)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); train_loss += float(loss.detach()); steps += 1
            val = run.score(model, validation)
            history.append({'epoch': epoch, 'train_objective': train_loss / steps, 'val_wave_mse': val})
            if val < best - 1e-7:
                best = val; stale = 0
                torch.save({'model': NAME, 'channel': channel, 'scale': amplitude,
                            'state_dict': model.state_dict(), 'epoch': epoch,
                            'lock_sha256': run.hashlib.sha256(run.LOCK.read_bytes()).hexdigest()}, file)
            else:
                stale += 1
                if stale >= run.PATIENCE:
                    break
        run.write_json(run.OUT / f'CH{1 if channel == 0 else 3}_{NAME}_history.json',
                       {'best_val_wave_mse': best, 'history': history,
                        'seconds': time.perf_counter() - began,
                        'note': '事后诊断，不能用此测试结果重新选择原 CNN / Bi-LSTM'})
        print(f'CH{1 if channel == 0 else 3} short-input CNN train best MSE={best:.8g}', flush=True)


def evaluate(lock):
    torch.set_num_threads(4)
    run.public.COMMON.init_worker()
    lock_hash = run.hashlib.sha256(run.LOCK.read_bytes()).hexdigest()
    networks = {}
    for channel in range(2):
        payload = torch.load(run.OUT / f'CH{1 if channel == 0 else 3}_{NAME}.pt',
                             map_location='cpu', weights_only=True)
        if payload['lock_sha256'] != lock_hash:
            raise ValueError('短路消融与冻结配置不一致')
        model = make_model('cnn'); model.load_state_dict(payload['state_dict']); model.eval()
        networks[channel] = (model, payload['scale'])
    result = np.empty((19, 3))
    for case in run.existing.CASES:
        file = run.OUT / f'case{case:02d}_{NAME}.json'
        if file.exists():
            row = json.loads(file.read_text(encoding='utf-8'))
            if row['lock_sha256'] == lock_hash:
                result[case - 1] = row['test']; continue
        symbols = []
        for channel in range(2):
            short, _ = run.full_pair(case, channel, lock)
            model, amplitude = networks[channel]
            fused = predict_wave(model, short, short, scale=amplitude)
            symbols.append(run.public.backend(fused, channel))
        ch = [float(run.public.rates(run.public.channel_counts(s, run.public.COMMON.REFERENCES, run.TEST), run.TEST)[0])
              for s in symbols]
        pbs = float(run.public.rates(run.public.weighted_counts(*symbols, np.array([.59]),
                                                              run.TEST, run.public.COMMON.REFERENCES), run.TEST)[0, 0])
        result[case - 1] = ch + [pbs]
        run.write_json(file, {'case': case, 'lock_sha256': lock_hash, 'test': result[case - 1].tolist()})
        print(f'case {case}/19 short-input CNN PBS={pbs:.8g}', flush=True)
    np.savetxt(run.OUT / f'{NAME}_test_berall.txt', result, fmt='%.16e')
    dual = np.loadtxt(run.OUT / 'cnn_test_berall.txt')
    short = np.loadtxt(run.public.NN / 'multiscale_quadratic_short_only_test_berall.txt')
    summary = {'mean': result.mean(axis=0).tolist(),
               'dual_cnn_mean': dual.mean(axis=0).tolist(),
               'short_mean': short.mean(axis=0).tolist(),
               'dual_minus_short_input': run.public.paired_statistics(dual[:, 2] - result[:, 2]),
               'short_input_minus_short': run.public.paired_statistics(result[:, 2] - short[:, 2]),
               'note': '事后诊断：该消融使用已观察过的严格测试，不用于重新选模型或宣称独立显著性'}
    run.write_json(run.OUT / 'ablation_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    locked = run.selected()
    train(locked)
    evaluate(locked)
