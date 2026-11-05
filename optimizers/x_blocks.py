"""Muon on blocks of a matrix: rows=R cuts it into R groups of rows, cols=C into C groups of columns, and each block is orthogonalized alone. rows=1, cols=1 is plain Muon and rows=4 is per-head Muon on a head-sized matrix. The whole step keeps plain Muon's size, so this changes only the shape of the step."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    beta = k.get("beta", .95); R, C = int(k.get("rows", 1)), int(k.get("cols", 1))
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(beta).add_(g); d = g + beta * st["b"]
    n, c = d.shape[0] // R, d.shape[1] // C; x = torch.zeros_like(d)
    for i in range(R):
        for j in range(C):
            x[i * n:(i + 1) * n, j * c:(j + 1) * c] = ns(d[i * n:(i + 1) * n, j * c:(j + 1) * c])
    w.add_(x * (d.shape[0] / (R * C * min(n, c))) ** .5, alpha=-lr)
