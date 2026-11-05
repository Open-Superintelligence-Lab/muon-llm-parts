"""Dion3 (NorDion2) from microsoft/dion nordion2.py: M += g; take the rows with the largest L1 norm, a fraction f of them; orthogonalize those rows; divide each by its running size, keeping the Frobenius norm; step those rows by lr sqrt(rows/cols); multiply only those rows of M by mu. No weight decay, like our other rules."""
import math
import torch


def ns(d):
  X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
  if tall: X = X.T
  for i in range(5):
    A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
  return X.T if tall else X


def step(w, g, st, lr, k):
  if w.ndim != 2: return
  f, mu, b2 = k.get("f", .25), k.get("beta", .95), k.get("b2", .95); m = st.setdefault("m", torch.zeros_like(g)); m.add_(torch.nan_to_num(g)); v = st.setdefault("v", torch.zeros(w.shape[0], 1, dtype=w.dtype, device=w.device))
  n = max(1, math.ceil(f * w.shape[0])); i = m.norm(p=1, dim=1).topk(n).indices; u = ns(m[i])
  v[i] = v[i] + (1 - b2) * ((u * u).mean(1, keepdim=True) - v[i]); y = u / (v[i].sqrt() + 1e-8); y = y * u.norm() / y.norm().clamp_min(1e-8)
  w[i] = w[i] - lr * (w.shape[0] / w.shape[1]) ** .5 * y; m[i] = m[i] * mu
