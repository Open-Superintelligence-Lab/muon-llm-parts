"""Half Muon across the whole matrix and half Muon inside each head. Whole-matrix Muon keeps heads apart, and per-head Muon gives each head an equal step. The step keeps Muon's size."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(.9).add_(g); d = g + .9 * st["b"]; h = int(k.get("h", 4)); n = w.shape[0] // h; z = ns(d).clone()
    for i in range(0, w.shape[0], n): z[i:i + n] = .5 * z[i:i + n] + .5 * ns(d[i:i + n])
    w.add_(z / (z.norm() + 1e-7) * w.shape[0] ** .5, alpha=-lr)
