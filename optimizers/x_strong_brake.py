"""Muon that sizes its strongest directions by Rprop's rule. Each strong rank has a size. It shrinks by down when the new gradient opposes that direction of the last step and grows by up otherwise, between floor and 1. Weak directions keep size 1. q is the share of directions watched."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    beta = k.get("beta", .95); h = int(k.get("h", 1)); n = w.shape[0] // h; share = k.get("q", .25); up, down, floor = k.get("up", 1.1), k.get("down", .7), k.get("floor", .1)
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(beta).add_(g); d = g + beta * st["b"]
    for j in range(h):
        a, c = j * n, (j + 1) * n; U, S, Vh = torch.linalg.svd(d[a:c], full_matrices=False); q = max(1, int(len(S) * share)); f = torch.ones_like(S)
        if f"u{j}" in st:
            v = (st[f"u{j}"].T @ g[a:c] @ st[f"v{j}"]).diagonal(); s = st[f"z{j}"] * torch.where(v < 0, torch.full_like(v, down), torch.full_like(v, up)); s = s.clamp(floor, 1); st[f"z{j}"] = s; f[:q] = s
        else: st[f"z{j}"] = torch.ones(q)
        st[f"u{j}"] = U[:, :q].clone(); st[f"v{j}"] = Vh[:q].T.clone()
        w[a:c].add_((ns(d[a:c]) - (U[:, :q] * (1 - f[:q])) @ Vh[:q]) * max(1.0, n / w.shape[1]) ** .5, alpha=-lr)
