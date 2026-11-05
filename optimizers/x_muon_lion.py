"""Half Muon, half Lion. Both directions are scaled to the same size, so one lr moves both."""
import torch


def step(w, g, st, lr, k):
    if "b" not in st: st["b"] = torch.zeros_like(w); st["m"] = torch.zeros_like(w)
    st["b"].mul_(.9).add_(g); st["m"].mul_(.9).add_(g, alpha=.1)
    X = g + .9 * st["b"]; X = X / (X.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(5):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    if tall: X = X.T
    big, small = max(w.shape), min(w.shape); w.add_(.5 * X * (big / small) ** .5 + .5 * st["m"].sign() / small ** .5, alpha=-lr)
