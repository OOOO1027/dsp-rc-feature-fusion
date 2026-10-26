"""冻结双 RC 波形后的局部 CNN / 双向 LSTM 残差融合器。"""
import numpy as np
import torch
from torch import nn

WINDOW = 256
HALO = 32
CORE = WINDOW - 2 * HALO
CORRECTION_SCALE = 0.25


class CNNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 12, 5, padding=2), nn.GELU(),
            nn.Conv1d(12, 12, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(12, 12, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(12, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        correction = self.net(x.transpose(1, 2)).transpose(1, 2)[..., 0]
        return x[..., 0] + CORRECTION_SCALE * torch.tanh(correction)


class BiLSTMFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=12, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.readout = nn.Linear(24, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x):
        state, _ = self.lstm(x)
        correction = self.readout(state)[..., 0]
        return x[..., 0] + CORRECTION_SCALE * torch.tanh(correction)


def make_model(name):
    if name == 'cnn':
        return CNNFusion()
    if name == 'bilstm':
        return BiLSTMFusion()
    raise ValueError(name)


def predict_wave(model, short, long, *, scale=1.0, batch_size=64):
    """分块推理；每块仅拼接中心 CORE 点，边界用 32 点上下文。"""
    short = np.asarray(short, dtype=np.float32).reshape(-1)
    long = np.asarray(long, dtype=np.float32).reshape(-1)
    if short.shape != long.shape or not len(short):
        raise ValueError('双分支长度必须相同且非空')
    if not np.isfinite(short).all() or not np.isfinite(long).all() or scale <= 0:
        raise ValueError('分支必须有限，训练尺度必须为正')
    x = np.column_stack((short / scale, long / scale))
    xpad = np.pad(x, ((HALO, HALO + CORE), (0, 0)), mode='edge')
    starts = np.arange(0, len(x), CORE)
    result = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            locations = starts[begin:begin + batch_size]
            batch = np.stack([xpad[int(at):int(at) + WINDOW] for at in locations])
            pred = model(torch.from_numpy(batch))[..., HALO:HALO + CORE].numpy()
            for at, row in zip(locations, pred):
                stop = min(int(at) + CORE, len(x))
                result[int(at):stop] = row[:stop - int(at)] * scale
    if not np.isfinite(result).all():
        raise FloatingPointError('神经融合输出非有限')
    return result.astype(np.float64)
