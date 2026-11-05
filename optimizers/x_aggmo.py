"""Control for x_weak_memory: two memories added together before plain orthogonalizing, with no split between strong and weak directions (Aggregated Momentum, Lucas et al. 2019). mix is the slow memory's share."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    beta = k.get("beta", .95); h = int(k.get("h", 1)); n = w.shape[0] // h; fast, slow, mix = k.get("fast", beta), k.get("slow", .98), k.get("mix", .5)
    if "f" not in st: st["f"] = torch.zeros_like(w); st["s"] = torch.zeros_like(w)
    st["f"].mul_(fast).add_(g, alpha=1 - fast); st["s"].mul_(slow).add_(g, alpha=1 - slow); d = g * (1 - fast) + fast * ((1 - mix) * st["f"] + mix * st["s"])
    for j in range(h): w[j * n:(j + 1) * n].add_(ns(d[j * n:(j + 1) * n]) * max(1.0, n / w.shape[1]) ** .5, alpha=-lr)
