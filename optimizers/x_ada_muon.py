"""Divides each number of the momentum by its gradient size, as Adam does, and then orthogonalizes. Muon then sees a direction not dominated by a few large numbers."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    if "b" not in st: st["b"] = torch.zeros_like(w); st["v"] = torch.zeros_like(w); st["t"] = 0
    st["t"] += 1; st["b"].mul_(.9).add_(g); st["v"].mul_(.99).addcmul_(g, g, value=.01); r = (st["v"] / (1 - .99 ** st["t"])).sqrt() + 1e-8
    w.add_(ns((g + .9 * st["b"]) / r), alpha=-lr * max(1, w.shape[0] / w.shape[1]) ** .5)
