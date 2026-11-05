"""Kaon from 2605.11181, Algorithm 2: Muon's backbone with the chaotic map X = 4.1 (I - XX^T)^2 X, 5 times, then divided by 1.175. Nesterov and scale as our Muon."""
import torch


def step(w, g, st, lr, k):
  if w.ndim != 2: return
  beta = k.get("beta", .95); b = st.setdefault("b", torch.zeros_like(g)); b.mul_(beta).add_(g); d = torch.nan_to_num(g + beta * b)
  X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
  if tall: X = X.T
  for i in range(5):
    A = X @ X.T; Y = X - A @ X; X = 4.1 * (Y - A @ Y)
  X = X / 1.175
  if tall: X = X.T
  w.add_(X * max(1.0, w.shape[0] / w.shape[1]) ** .5, alpha=-lr)
