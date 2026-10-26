"""Supplementary, post-hoc multiseed comparison of locked dual/short RC CNNs.

All training uses the existing frozen RC, train/validation boundaries, and
physical PBS weight. The strict test is reported, never used for checkpoint
selection. These runs are exploratory because earlier results were seen.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import rx_neural_dual_rc_lms as base
from neural_dual_rc import HALO, WINDOW, make_model, predict_wave

SEEDS = (20261025, 20261026, 20261027, 20261028, 20261029)
VARIANTS = ('dual', 'short')
OUT = base.OUT / 'multiseed_core'


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    base.public.COMMON.init_worker()
    lock = base.selected()
    lock_hash = hashlib.sha256(base.LOCK.read_bytes()).hexdigest()
    amplitude = base.scale()
    for channel in range(2):
        tx0, ty, vx0, vy = base.assemble(channel, lock, amplitude)
        for variant in VARIANTS:
            tx, vx = tx0.clone(), vx0.clone()
            if variant == 'short':
                tx[:, :, 1] = tx[:, :, 0]
                vx[:, :, 1] = vx[:, :, 0]
            val_loader = DataLoader(TensorDataset(vx, vy), batch_size=64, shuffle=False)
            for seed in SEEDS[1:]:
                stem = f's{seed}_CH{1 if channel == 0 else 3}_{variant}'
                model_file = OUT / f'{stem}.pt'
                history_file = OUT / f'{stem}_history.json'
                if model_file.exists() and history_file.exists():
                    print('cached train', stem, flush=True)
                    continue
                torch.manual_seed(seed)
                np.random.seed(seed)
                loader = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=True,
                                    generator=torch.Generator().manual_seed(seed + channel))
                model = make_model('cnn')
                optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-5)
                best = float('inf')
                stale = 0
                history = []
                began = time.perf_counter()
                for epoch in range(1, base.EPOCHS + 1):
                    model.train()
                    total = 0.0
                    count = 0
                    for x, y in loader:
                        pred = model(x)[:, HALO:WINDOW - HALO]
                        short = x[:, HALO:WINDOW - HALO, 0]
                        loss = torch.mean((pred - y) ** 2) + .02 * torch.mean((pred - short) ** 2)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        total += float(loss.detach())
                        count += 1
                    validation = base.score(model, val_loader)
                    history.append({'epoch': epoch, 'train_objective': total / count,
                                    'val_wave_mse': validation})
                    if validation < best - 1e-7:
                        best = validation
                        stale = 0
                        torch.save({'state_dict': model.state_dict(), 'seed': seed, 'variant': variant,
                                    'channel': channel, 'scale': amplitude, 'epoch': epoch,
                                    'lock_sha256': lock_hash}, model_file)
                    else:
                        stale += 1
                        if stale >= base.PATIENCE:
                            break
                base.write_json(history_file, {'seed': seed, 'variant': variant, 'channel': channel,
                                               'best_val_wave_mse': best, 'history': history,
                                               'seconds': time.perf_counter() - began})
                print('trained', stem, 'epoch', min(history,key=lambda a:a['val_wave_mse'])['epoch'],
                      'val', f'{best:.6g}', flush=True)


def load_networks():
    lock_hash = hashlib.sha256(base.LOCK.read_bytes()).hexdigest()
    loaded = {}
    for seed in SEEDS[1:]:
        for channel in range(2):
            for variant in VARIANTS:
                path = OUT / f's{seed}_CH{1 if channel == 0 else 3}_{variant}.pt'
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if (payload['lock_sha256'] != lock_hash or payload['seed'] != seed
                        or payload['channel'] != channel or payload['variant'] != variant):
                    raise ValueError(f'model mismatch: {path}')
                model = make_model('cnn')
                model.load_state_dict(payload['state_dict'])
                model.eval()
                loaded[(seed, channel, variant)] = (model, float(payload['scale']))
    return loaded


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    base.public.COMMON.init_worker()
    lock = base.selected()
    lock_hash = hashlib.sha256(base.LOCK.read_bytes()).hexdigest()
    networks = load_networks()
    # Existing seed is reused with the original, identical setup.
    result = {}
    for variant, name in [('dual','cnn'),('short','cnn_short_input')]:
        result[(SEEDS[0],variant)] = np.loadtxt(base.OUT / f'{name}_test_berall.txt')
    for seed in SEEDS[1:]:
        for variant in VARIANTS:
            result[(seed,variant)] = np.full((19,3), np.nan)
    for case in base.existing.CASES:
        cache = OUT / f'case{case:02d}.json'
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved.get('lock_sha256') == lock_hash:
                for key, value in saved['test'].items():
                    seed,variant = key.split('_',1)
                    result[(int(seed),variant)][case-1] = value
                print('cached eval case',case,flush=True)
                continue
        symbol = {(seed,variant):[] for seed in SEEDS[1:] for variant in VARIANTS}
        for channel in range(2):
            short,long = base.full_pair(case, channel, lock)
            for seed in SEEDS[1:]:
                for variant in VARIANTS:
                    model,amplitude = networks[(seed,channel,variant)]
                    wave = predict_wave(model, short, long if variant=='dual' else short,
                                        scale=amplitude)
                    symbol[(seed,variant)].append(base.public.backend(wave, channel))
        saved={'case':case,'lock_sha256':lock_hash,'test':{}}
        for (seed,variant), pair in symbol.items():
            ch=[float(base.public.rates(base.public.channel_counts(s,base.public.COMMON.REFERENCES,base.TEST),base.TEST)[0]) for s in pair]
            pbs=float(base.public.rates(base.public.weighted_counts(*pair,np.array([.59]),base.TEST,base.public.COMMON.REFERENCES),base.TEST)[0,0])
            row=ch+[pbs]
            result[(seed,variant)][case-1]=row
            saved['test'][f'{seed}_{variant}']=row
        base.write_json(cache,saved)
        print('evaluated case',case,'/',19,flush=True)
    rows=[]
    for seed in SEEDS:
        dual=result[(seed,'dual')]
        short=result[(seed,'short')]
        if not np.isfinite(dual).all() or not np.isfinite(short).all():
            raise ValueError(f'incomplete results for {seed}')
        for variant,array in [('dual',dual),('short',short)]:
            np.savetxt(OUT/f's{seed}_{variant}_test_berall.txt',array,fmt='%.16e')
            rows.append({'seed':seed,'variant':variant,'CH1':float(array[:,0].mean()),
                         'CH3':float(array[:,1].mean()),'PBS':float(array[:,2].mean())})
        diff=dual[:,2]-short[:,2]
        print('seed',seed,'dual PBS',dual[:,2].mean(),'short PBS',short[:,2].mean(),
              'paired delta',diff.mean(),flush=True)
    summary={'note':'Post-hoc exploratory multiseed analysis; same 19 strict-test groups and frozen PBS weight 0.59. Seed 20261025 reuses original results.',
             'rows':rows,
             'mean_across_seeds':{v:float(np.mean([r['PBS'] for r in rows if r['variant']==v])) for v in VARIANTS},
             'paired_seed_differences':[float((result[(s,'dual')][:,2]-result[(s,'short')][:,2]).mean()) for s in SEEDS],
             'dual_wins_seeds':sum(float((result[(s,'dual')][:,2]-result[(s,'short')][:,2]).mean())<0 for s in SEEDS)}
    base.write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    train()
    evaluate()
