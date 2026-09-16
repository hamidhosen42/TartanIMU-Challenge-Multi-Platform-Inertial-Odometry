"""Unified context model for multi-platform inertial odometry.

Input : raw 200 Hz IMU chunk (B, 6, L) covering T = L/200 consecutive windows of one trajectory.
Output: dense body-frame velocity at 20 Hz (B, T*20, 3) -> averaged to one velocity per 1 s window,
        plus platform logits (auxiliary supervision only; never used for routing).

Architecture: strided conv stem (200 Hz -> 20 Hz tokens) -> dilated depthwise TCN (~13 s receptive
field) -> 2-layer transformer encoder (whole-chunk context, past and future) -> per-token velocity head.
One shared set of weights handles all four embodiments.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import WIN

TOK = 20                                  # tokens per 1 s window (200 Hz / 10)
CHUNK_WIN = 16                            # windows per training / inference chunk
# fixed input scaling (raw units -> O(1)); zero mean keeps rotation augmentation exact
IN_SCALE = torch.tensor([3.0, 3.0, 3.0, 0.4, 0.4, 0.4]).view(1, 6, 1)


class TCNBlock(nn.Module):
    def __init__(self, w: int, dilation: int, drop: float = 0.1):
        super().__init__()
        self.norm = nn.GroupNorm(8, w)
        self.dw = nn.Conv1d(w, w, 5, padding=2 * dilation, dilation=dilation, groups=w, bias=False)
        self.pw = nn.Sequential(nn.Conv1d(w, 3 * w, 1), nn.GELU(), nn.Dropout(drop), nn.Conv1d(3 * w, w, 1))

    def forward(self, x):
        return x + self.pw(self.dw(self.norm(x)))


class IMUNet(nn.Module):
    def __init__(self, width: int = 128, blocks: int = 8, ctx_layers: int = 2, drop: float = 0.1, max_tok: int = 4096):
        super().__init__()
        self.register_buffer("in_scale", IN_SCALE.clone())
        self.stem = nn.Sequential(
            nn.Conv1d(6, width // 2, 9, stride=2, padding=4, bias=False), nn.GroupNorm(8, width // 2), nn.GELU(),
            nn.Conv1d(width // 2, width, 11, stride=5, padding=5, bias=False), nn.GroupNorm(8, width), nn.GELU())
        dil = (1, 2, 4, 8, 16, 32, 1, 2)
        self.tcn = nn.Sequential(*[TCNBlock(width, dil[i % len(dil)], drop) for i in range(blocks)])
        self.pos = nn.Parameter(torch.zeros(1, max_tok, width)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, 4, 3 * width, dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        self.ctx = nn.TransformerEncoder(layer, ctx_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 3))
        self.attn = nn.Linear(width, 1)
        self.plat = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 4))

    def forward(self, x):                         # x: (B, 6, L) raw IMU
        h = self.tcn(self.stem(x / self.in_scale))  # (B, W, L/10)
        h = h.transpose(1, 2)                       # (B, N, W)
        h = self.ctx(h + self.pos[:, : h.shape[1]])
        dense = self.head(h)                        # (B, N, 3)  20 Hz velocity
        a = torch.softmax(self.attn(h), dim=1)
        plat = self.plat(torch.cat([(h * a).sum(1), h.mean(1)], dim=-1))
        return dense, plat

    @staticmethod
    def to_windows(dense):                          # (B, N, 3) -> (B, N/TOK, 3)
        B, N, _ = dense.shape
        return dense.view(B, N // TOK, TOK, 3).mean(2)


# --------------------------------------------------------------------------- trajectory inference
@torch.no_grad()
def predict_trajectory(model: nn.Module, imu: np.ndarray, device, chunk_win: int = CHUNK_WIN, stride_win: int = 4,
                       batch: int = 32) -> np.ndarray:
    """Sliding-chunk inference over one whole trajectory. Returns (n_win, 3) window velocities.

    Overlapping chunks are averaged with a raised-cosine weight so each window is trusted most from the
    chunk where it sits near the centre (full bidirectional context). Short trajectories are edge-padded.
    """
    n_win = len(imu) // WIN
    x = imu[: n_win * WIN].astype(np.float32)
    pad = max(0, chunk_win - n_win)
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad * WIN, axis=0)])
    total = n_win + pad
    starts = list(range(0, total - chunk_win + 1, stride_win))
    if starts[-1] != total - chunk_win:
        starts.append(total - chunk_win)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(chunk_win) + 0.5) / chunk_win)   # Hann, >0 everywhere
    w = w.astype(np.float32) + 0.05
    acc = np.zeros((total, 3), np.float32)
    wsum = np.zeros((total, 1), np.float32)
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        xb = np.stack([x[s * WIN:(s + chunk_win) * WIN].T for s in ss])
        dense, _ = model(torch.from_numpy(xb).to(device))
        vb = model.to_windows(dense).float().cpu().numpy()
        for s, v in zip(ss, vb):
            acc[s:s + chunk_win] += v * w[:, None]
            wsum[s:s + chunk_win] += w[:, None]
    return (acc / wsum)[:n_win]
