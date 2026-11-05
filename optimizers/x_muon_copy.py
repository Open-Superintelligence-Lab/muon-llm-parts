"""Muon written as an invented optimizer: the example to copy, and the check that the sandbox changes nothing."""
import torch


def step(w, g, st, lr, k):
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(.95).add_(g); d = g + .95 * st["b"]; X = d / (d.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(5):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    if tall: X = X.T
    w.add_(X, alpha=-lr * max(1, w.shape[0] / w.shape[1]) ** .5)
