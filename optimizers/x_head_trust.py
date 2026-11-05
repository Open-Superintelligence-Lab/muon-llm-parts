"""Per-head Muon that moves a head further when its gradient keeps agreeing with its momentum. A head whose gradients look like noise moves less. The total step keeps Muon's size."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    h = int(k.get("h", 4)); n = w.shape[0] // h
    if "b" not in st: st["b"] = torch.zeros_like(w); st["u"] = torch.zeros(h)
    for j in range(h):
        x, y = g[j * n:(j + 1) * n], st["b"][j * n:(j + 1) * n]; st["u"][j] = .95 * st["u"][j] + .05 * (x * y).sum() / (x.norm() * y.norm() + 1e-12)
    s = st["u"].clamp_min(0) + .1; s = s * (h / (s * s).sum()) ** .5; st["b"].mul_(.9).add_(g); d = g + .9 * st["b"]
    for j in range(h): w[j * n:(j + 1) * n].add_(ns(d[j * n:(j + 1) * n]), alpha=-lr * float(s[j]))
