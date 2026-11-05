"""Orthogonalizes each gradient first and then averages, the reverse of Muon's order. One noisy step cannot steer the whole direction, and the step shrinks when recent gradients disagree."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    if "m" not in st: st["m"] = torch.zeros_like(w)
    st["m"].mul_(.9).add_(ns(g), alpha=.1); w.add_(st["m"], alpha=-lr * max(1, w.shape[0] / w.shape[1]) ** .5)
