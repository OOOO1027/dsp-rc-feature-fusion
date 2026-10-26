"""Post-hoc no-skip CNN ablation on frozen short/long RC waveforms."""
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import rx_neural_dual_rc_lms as base
from neural_dual_rc import HALO, WINDOW, CNNFusion, predict_wave

OUT = base.OUT / 'nores_core'
SEED = base.SEED


class CNNNoSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = CNNFusion().net
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)

    def forward(self,x):
        return self.net(x.transpose(1,2)).transpose(1,2)[...,0]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base.public.COMMON.init_worker()
    lock=base.selected()
    lock_hash=hashlib.sha256(base.LOCK.read_bytes()).hexdigest()
    amplitude=base.scale()
    models={}
    for channel in range(2):
        file=OUT/f'CH{1 if channel==0 else 3}_cnn_no_skip.pt'
        history_file=OUT/f'CH{1 if channel==0 else 3}_cnn_no_skip_history.json'
        if not file.exists() or not history_file.exists():
            tx,ty,vx,vy=base.assemble(channel,lock,amplitude)
            train_loader=DataLoader(TensorDataset(tx,ty),batch_size=32,shuffle=True,
                                    generator=torch.Generator().manual_seed(SEED+channel))
            val_loader=DataLoader(TensorDataset(vx,vy),batch_size=64,shuffle=False)
            model=CNNNoSkip()
            opt=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-5)
            best=float('inf'); stale=0; history=[]; began=time.perf_counter()
            for epoch in range(1,base.EPOCHS+1):
                model.train(); total=0.; steps=0
                for x,y in train_loader:
                    pred=model(x)[:,HALO:WINDOW-HALO]
                    short=x[:,HALO:WINDOW-HALO,0]
                    loss=torch.mean((pred-y)**2)+.02*torch.mean((pred-short)**2)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    opt.step(); total+=float(loss.detach()); steps+=1
                val=base.score(model,val_loader)
                history.append({'epoch':epoch,'train_objective':total/steps,'val_wave_mse':val})
                if val<best-1e-7:
                    best=val; stale=0
                    torch.save({'state_dict':model.state_dict(),'channel':channel,'scale':amplitude,
                                'epoch':epoch,'lock_sha256':lock_hash},file)
                else:
                    stale+=1
                    if stale>=base.PATIENCE: break
            base.write_json(history_file,{'channel':channel,'best_val_wave_mse':best,
                                          'history':history,'seconds':time.perf_counter()-began,
                                          'parameter_count':sum(p.numel() for p in model.parameters())})
            print('trained no-skip CH',1 if channel==0 else 3,'val',best,flush=True)
        payload=torch.load(file,map_location='cpu',weights_only=True)
        if payload['lock_sha256']!=lock_hash: raise ValueError('lock mismatch')
        model=CNNNoSkip(); model.load_state_dict(payload['state_dict']); model.eval()
        models[channel]=(model,float(payload['scale']))
    result=np.full((19,3),np.nan)
    for case in base.existing.CASES:
        cache=OUT/f'case{case:02d}.json'
        if cache.exists():
            saved=json.loads(cache.read_text(encoding='utf-8'))
            if saved.get('lock_sha256')==lock_hash:
                result[case-1]=saved['test']; continue
        pair=[]
        for channel in range(2):
            short,long=base.full_pair(case,channel,lock)
            model,scale=models[channel]
            wave=predict_wave(model,short,long,scale=scale)
            pair.append(base.public.backend(wave,channel))
        ch=[float(base.public.rates(base.public.channel_counts(s,base.public.COMMON.REFERENCES,base.TEST),base.TEST)[0]) for s in pair]
        pbs=float(base.public.rates(base.public.weighted_counts(*pair,np.array([.59]),base.TEST,base.public.COMMON.REFERENCES),base.TEST)[0,0])
        result[case-1]=ch+[pbs]
        base.write_json(cache,{'case':case,'lock_sha256':lock_hash,'test':result[case-1].tolist()})
        print('evaluated no-skip',case,'/',19,flush=True)
    np.savetxt(OUT/'cnn_no_skip_test_berall.txt',result,fmt='%.16e')
    reference=np.loadtxt(base.OUT/'cnn_test_berall.txt')
    summary={'mean':result.mean(axis=0).tolist(),
             'reference_cnn_dual_residual_mean':reference.mean(axis=0).tolist(),
             'no_skip_minus_residual':base.public.paired_statistics(result[:,2]-reference[:,2]),
             'method':'Same 1609-parameter 3-layer convolution stack, data, train/validation split, MSE + 0.02 short-correction penalty and PBS=0.59. No-skip output is unconstrained; residual output uses 0.25*tanh correction.',
             'limitation':'Post-hoc structural comparison; no new blind test.'}
    base.write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
