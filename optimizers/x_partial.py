"""Muon with 2 orthogonalizing steps in place of 5. Large directions reach full size, and small noisy ones stay small instead of being raised to the same size. The step is rescaled to Muon's size, and it costs less time."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(.9).add_(g); z = ns(g + .9 * st["b"], 2); w.add_(z / (z.norm() + 1e-7) * min(w.shape) ** .5, alpha=-lr * max(1, w.shape[0] / w.shape[1]) ** .5)
