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

WIN = 200  # samples per 1 s window (200 Hz)

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


def _exp_so3(w):
    """Batched Rodrigues: rotation vectors (..., 3) -> matrices (..., 3, 3)."""
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    st, ct = th.sin()[..., None], th.cos()[..., None]
    return I + st * K + (1 - ct) * (K @ K)


@torch.no_grad()
def physics_features(x, dt_tok: float = TOK / WIN / TOK * 10):
    """Deterministic strap-down features at token rate from a raw chunk x (B, 6, L).

    Gyro is integrated (parallel prefix product of per-token rotations) to get R_t: body(t) -> body(chunk start).
    Accelerometer is rotated into that common frame, its chunk mean is taken as the gravity estimate, and the
    de-gravitated acceleration is integrated to a relative velocity.  Everything is then expressed back in the
    *current* body frame, so all outputs are ordinary body-frame vectors (rotation-equivariant like the target):
      a_dyn (3): accelerometer minus estimated gravity,   g_b (3): estimated gravity direction in body frame,
      dv (3):    velocity change since chunk start.       Returns (B, 9, N) with N = L/10 tokens.
    """
    B, _, L = x.shape
    N = L // 10
    xb = x.float().view(B, 6, N, 10).mean(-1)                       # 20 Hz
    acc, gyr = xb[:, :3].transpose(1, 2), xb[:, 3:].transpose(1, 2)  # (B, N, 3)
    dt = 10.0 / 200.0
    E = _exp_so3(gyr * dt)                                          # per-token increments
    P = E.clone(); d = 1                                            # Hillis-Steele scan: P[t] = E_1 ... E_t
    while d < N:
        P = torch.cat([P[:, :d], P[:, :-d] @ P[:, d:]], dim=1); d *= 2
    R = P                                                           # (B, N, 3, 3): body(t) -> body(0)
    a0 = (R @ acc[..., None])[..., 0]                               # accel in start frame
    g0 = a0.mean(1, keepdim=True)                                   # gravity estimate (start frame)
    v0 = torch.cumsum(a0 - g0, dim=1) * dt                          # relative velocity (start frame)
    Rt = R.transpose(-1, -2)
    dv = (Rt @ v0[..., None])[..., 0]
    gb = (Rt @ g0.expand_as(a0)[..., None])[..., 0]
    a_dyn = acc - gb
    return torch.cat([a_dyn / 3.0, gb / 9.81, dv / 3.0], dim=-1).transpose(1, 2)  # (B, 9, N)


class IMUNet(nn.Module):
    def __init__(self, width: int = 128, blocks: int = 8, ctx_layers: int = 2, drop: float = 0.1, max_tok: int = 4096,
                 physics: bool = False, lag: bool = False):
        super().__init__()
        self.physics, self.lag = physics, lag
        self.register_buffer("in_scale", IN_SCALE.clone())
        if physics:
            self.phys_proj = nn.Sequential(nn.Conv1d(18, width, 1), nn.GELU(), nn.Conv1d(width, width, 1))
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
        if lag:   # per-chunk IMU-vs-ground-truth time offset (in 50 ms tokens); some recordings are offset by up to 70 ms
            self.lag_head = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.lag_head[-1].weight); nn.init.zeros_(self.lag_head[-1].bias)

    @staticmethod
    def shift_dense(dense, delta):
        """Fractional time shift of a (B, N, 3) sequence: out[t] = dense[t + delta], delta (B,) in tokens, edge-clamped."""
        B, N, _ = dense.shape
        pos = torch.arange(N, device=dense.device, dtype=dense.dtype)[None] + delta[:, None]      # (B, N)
        pos = pos.clamp(0, N - 1)
        i0 = pos.floor().long(); i1 = (i0 + 1).clamp(max=N - 1); w = (pos - i0.to(dense.dtype))[..., None]
        g0 = torch.gather(dense, 1, i0[..., None].expand(B, N, 3)); g1 = torch.gather(dense, 1, i1[..., None].expand(B, N, 3))
        return g0 * (1 - w) + g1 * w

    def forward(self, x, return_lag: bool = False):                         # x: (B, 6, L) raw IMU
        h = self.stem(x / self.in_scale)            # (B, W, L/10)
        if self.physics:
            # two hypotheses for the gyro-z sign (one drone source has it inverted); the network learns which to trust
            xz = torch.cat([x[:, :5], -x[:, 5:6]], dim=1)
            h = h + self.phys_proj(torch.cat([physics_features(x), physics_features(xz)], dim=1))
        h = self.tcn(h)
        h = h.transpose(1, 2)                       # (B, N, W)
        h = self.ctx(h + self.pos[:, : h.shape[1]])
        dense = self.head(h)                        # (B, N, 3)  20 Hz velocity
        a = torch.softmax(self.attn(h), dim=1)
        desc = torch.cat([(h * a).sum(1), h.mean(1)], dim=-1)
        plat = self.plat(desc)
        delta = None
        if self.lag:
            delta = 2.0 * torch.tanh(self.lag_head(desc)[:, 0])                # +-2 tokens = +-100 ms
            dense = self.shift_dense(dense, delta)
        if return_lag:
            return dense, plat, delta
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
