"""Per-head Muon that removes the part of a head's step that undoes its previous step, so a head does not swing back and forth. The step keeps Muon's size."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    h = int(k.get("h", 4)); n = w.shape[0] // h
    if "b" not in st: st["b"] = torch.zeros_like(w); st["p"] = torch.zeros_like(w)
    st["b"].mul_(.9).add_(g); d = g + .9 * st["b"]
    for j in range(h):
        p = st["p"][j * n:(j + 1) * n]; z = ns(d[j * n:(j + 1) * n]); c = (z * p).sum() / ((p * p).sum() + 1e-12)
        if c < 0: z = z - c * p
        z = z / (z.norm() + 1e-7) * n ** .5; st["p"][j * n:(j + 1) * n] = z; w[j * n:(j + 1) * n].add_(z, alpha=-lr)
