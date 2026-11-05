"""Muon with a fast and a slow momentum added together, so it follows recent gradients and keeps a long memory."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    if "a" not in st: st["a"] = torch.zeros_like(w); st["b"] = torch.zeros_like(w)
    st["a"].mul_(.7).add_(g, alpha=.3); st["b"].mul_(.98).add_(g, alpha=.02); w.add_(ns(g + 2 * st["a"] + 8 * st["b"]), alpha=-lr * max(1, w.shape[0] / w.shape[1]) ** .5)
