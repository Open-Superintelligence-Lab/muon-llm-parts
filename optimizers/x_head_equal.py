"""x_head_trust with the trust part removed, so every head takes an equal step. It is the control that shows what trust adds."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    h = int(k.get("h", 4)); n = w.shape[0] // h
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(.9).add_(g); d = g + .9 * st["b"]
    for j in range(h): w[j * n:(j + 1) * n].add_(ns(d[j * n:(j + 1) * n]), alpha=-lr)
