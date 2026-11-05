"""Muon with n Newton-Schulz steps."""
import torch


def step(w, g, st, lr, k):
  if w.ndim != 2: return
  beta = k.get("beta", .95); b = st.setdefault("b", torch.zeros_like(g)); b.mul_(beta).add_(g); d = torch.nan_to_num(g + beta * b)
  X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
  if tall: X = X.T
  for i in range(int(k.get("n", 5))):
    A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
  if tall: X = X.T
  w.add_(X * max(1.0, w.shape[0] / w.shape[1]) ** .5, alpha=-lr)
