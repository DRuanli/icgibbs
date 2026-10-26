"""
cell_pfn.py — một PFN nhỏ huấn luyện trên đúng prior ô–Dirichlet, dùng làm "TFM xấp xỉ" để kiểm chứng
Định lý 2 (q học được, sai số ε thật, các xác suất có điều kiện không tương thích một cách tự nhiên).

Kiến trúc: tập token ngữ cảnh h_j = E_ô[c_j] + E_nhãn[y_j]; token truy vấn h_* = E_ô[c_*] + E_nhãn["?"].
Các token ngữ cảnh tự-attention; truy vấn attend tới {chính nó, ngữ cảnh} (chính nó đóng vai pseudo-count,
xem coin_pfn.py trong khóa học). Huấn luyện bằng cross-entropy trên prior (định lý PFN ⇒ nghiệm tối ưu = PPD).

    python cell_pfn.py --steps 6000 --out ../../results/cell_pfn      (CPU, vài phút)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icgibbs.conditional import ConditionalModel  # noqa: E402


class Block(nn.Module):
    def __init__(self, E, H):
        super().__init__()
        self.att_ctx = nn.MultiheadAttention(E, H, batch_first=True)
        self.att_q = nn.MultiheadAttention(E, H, batch_first=True)
        self.ff_ctx = nn.Sequential(nn.Linear(E, 2 * E), nn.GELU(), nn.Linear(2 * E, E))
        self.ff_q = nn.Sequential(nn.Linear(E, 2 * E), nn.GELU(), nn.Linear(2 * E, E))
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(E) for _ in range(4))

    def forward(self, hc, hq, mask_c):
        # hc: (B, n, E) ngữ cảnh; hq: (B, Q, E) truy vấn; mask_c: (B, n) True = vị trí đệm
        if hc.shape[1] > 0:
            a = self.att_ctx(hc, hc, hc, key_padding_mask=mask_c)[0]
            hc = self.n1(hc + a)
            hc = self.n2(hc + self.ff_ctx(hc))
        B, Q, E = hq.shape
        keys = torch.cat([hq.reshape(B * Q, 1, E), hc.repeat_interleave(Q, 0)], 1)
        kmask = torch.cat([torch.zeros(B * Q, 1, dtype=torch.bool), mask_c.repeat_interleave(Q, 0)], 1)
        a = self.att_q(hq.reshape(B * Q, 1, E), keys, keys, key_padding_mask=kmask)[0].reshape(B, Q, E)
        hq = self.n3(hq + a)
        hq = self.n4(hq + self.ff_q(hq))
        return hc, hq


class CellPFN(nn.Module):
    def __init__(self, n_cells=3, n_classes=2, E=64, H=4, L=2):
        super().__init__()
        self.C = n_classes
        self.cell = nn.Embedding(n_cells, E)
        self.lab = nn.Embedding(n_classes + 1, E)
        self.blocks = nn.ModuleList([Block(E, H) for _ in range(L)])
        self.out = nn.Linear(E, n_classes)
        self.null = nn.Parameter(torch.zeros(1, 1, E))

    def forward(self, c_ctx, y_ctx, mask_c, c_q):
        B = c_q.shape[0]
        hc = self.cell(c_ctx) + self.lab(y_ctx)
        hc = torch.cat([self.null.expand(B, 1, -1), hc], 1)          # token rỗng luôn có mặt: tránh hàng bị che hết
        mask_c = torch.cat([torch.zeros(B, 1, dtype=torch.bool), mask_c], 1)
        hq = self.cell(c_q) + self.lab(torch.full_like(c_q, self.C))
        for b in self.blocks:
            hc, hq = b(hc, hq, mask_c)
        return self.out(hq)


def prior_batch(B, n_max, M, C, alpha, rng):
    n = rng.integers(0, n_max + 1, size=B)
    N = int(n.max())
    theta = rng.dirichlet(alpha * np.ones(C), size=(B, M))                  # (B, M, C)
    cells = rng.integers(0, M, size=(B, N + 1))
    u = rng.uniform(size=(B, N + 1, 1))
    cum = np.take_along_axis(theta, cells[..., None].repeat(C, -1), 1).cumsum(-1)
    y = (cum < u).sum(-1).clip(max=C - 1)
    mask = np.arange(N)[None] >= n[:, None]
    return (torch.tensor(cells[:, :N]), torch.tensor(y[:, :N]), torch.tensor(mask),
            torch.tensor(cells[:, N:]), torch.tensor(y[:, N]))


def train(steps, M=3, C=2, alpha=1.0, n_max=13, B=256, lr=1e-3, seed=0, snapshots=(), out=None, log=500):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CellPFN(M, C)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    saved = {}
    for step in range(1, steps + 1):
        c, y, m, cq, yq = prior_batch(B, n_max, M, C, alpha, rng)
        logits = model(c, y, m, cq)[:, 0]
        loss = nn.functional.cross_entropy(logits, yq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
        if step in snapshots and out:
            path = f"{out}_step{step}.pt"
            torch.save(model.state_dict(), path)
            saved[step] = path
    if out:
        torch.save(model.state_dict(), f"{out}_final.pt")
    return model, saved


class CellPFNConditional(ConditionalModel):
    """Bọc CellPFN thành ConditionalModel; có thêm batch_predict cho nhiều ngữ cảnh cùng lúc."""

    def __init__(self, model: CellPFN, n_classes=2):
        super().__init__(n_classes, floor=0.0)
        self.model = model.eval()

    def _predict(self, X_ctx, y_ctx, X_query):
        c = torch.tensor(X_ctx[:, 0].astype(int))[None]
        y = torch.tensor(y_ctx)[None]
        m = torch.zeros(1, len(y_ctx), dtype=torch.bool)
        cq = torch.tensor(X_query[:, 0].astype(int))[None]
        with torch.no_grad():
            return torch.softmax(self.model(c, y, m, cq)[0], -1).numpy().astype(float)

    def batch_predict(self, cells_ctx: np.ndarray, y_ctx: np.ndarray, cells_q: np.ndarray, chunk=4096):
        """cells_ctx, y_ctx: (B, n); cells_q: (B,) → (B, C). Mọi ngữ cảnh cùng kích thước n."""
        out = []
        with torch.no_grad():
            for s in range(0, len(cells_q), chunk):
                c = torch.tensor(cells_ctx[s:s + chunk])
                y = torch.tensor(y_ctx[s:s + chunk])
                m = torch.zeros(c.shape, dtype=torch.bool)
                cq = torch.tensor(cells_q[s:s + chunk])[:, None]
                out.append(torch.softmax(self.model(c, y, m, cq)[:, 0], -1).numpy())
        self.n_calls += len(cells_q)
        return np.concatenate(out).astype(float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--snapshots", default="100,300,1000,3000")
    ap.add_argument("--out", default="cell_pfn")
    a = ap.parse_args()
    snaps = tuple(int(s) for s in a.snapshots.split(",") if s)
    train(a.steps, snapshots=snaps, out=a.out)
