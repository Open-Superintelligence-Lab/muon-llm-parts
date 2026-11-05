"""Per-head Muon with a small push that keeps heads apart. Each head also moves away from what the other heads' rows already read, so heads should not learn the same thing. The total step of a head keeps Muon's size."""
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
    st["b"].mul_(.9).add_(g); d = g + .9 * st["b"]; old = w.clone()
    for j in range(h):
        o = old.clone(); o[j * n:(j + 1) * n] = 0; a = old[j * n:(j + 1) * n] @ (o.T @ o); z = ns(d[j * n:(j + 1) * n]); z = z + .1 * a / (a.norm() + 1e-7) * z.norm()
        w[j * n:(j + 1) * n].add_(z / (z.norm() + 1e-7) * n ** .5, alpha=-lr)
