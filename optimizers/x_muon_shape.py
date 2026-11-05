"""Muon with its step sizes set per singular direction. top scales the strongest quarter of the directions and weak scales the rest. Without either knob this is plain Muon (h=1) or per-head Muon (h=4), with both at 1 it is exact orthogonalization by SVD. shift=1 pairs each weak left direction with the wrong right direction, so the weak part keeps its size and loses the gradient's information."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def shaped(d, top, weak, shift):
    U, S, V = torch.linalg.svd(d, full_matrices=False); q = max(1, len(S) // 4); f = torch.full_like(S, weak); f[:q] = top
    if shift: V = torch.cat((V[:q], V[q:].roll(1, 0)))
    return (U * f) @ V


def step(w, g, st, lr, k):
    beta = k.get("beta", .95); h = int(k.get("h", 1)); n = w.shape[0] // h
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(beta).add_(g); d = g + beta * st["b"]
    for j in range(h):
        a, c = j * n, (j + 1) * n; x = shaped(d[a:c], k.get("top", 1), k.get("weak", 1), k.get("shift", 0)) if "top" in k or "weak" in k or "shift" in k else ns(d[a:c])
        w[a:c].add_(x * max(1.0, n / w.shape[1]) ** .5, alpha=-lr)
