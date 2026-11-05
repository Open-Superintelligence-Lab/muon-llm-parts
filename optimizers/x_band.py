"""Polar update with a set step per quarter of singular directions. Takes the exact SVD of the Nesterov momentum d = U S V^T, sorts directions by singular value, and steps U diag(m) V^T * sqrt(rows / rank), where m is t1 for the top quarter, t2, t3 and t4 for the next ones. All t equal to 1 is exact polar at Muon's update size. Used to ask which quarter of the spectrum carries the exponent's gain."""
import torch


def step(w, g, st, lr, k):
  if w.ndim != 2: return
  beta = k.get("beta", .95)
  b = st.setdefault("b", torch.zeros_like(g)); b.mul_(beta).add_(g); d = torch.nan_to_num(g + beta * b)
  U, S, V = torch.linalg.svd(d, full_matrices=False); r = S.shape[0]
  m = torch.ones_like(S); q = torch.arange(r, device=S.device) * 4 // r
  m[q == 0] = k.get("t1", 1.0); m[q == 1] = k.get("t2", 1.0); m[q == 2] = k.get("t3", 1.0); m[q == 3] = k.get("t4", 1.0)
  u = (U * m) @ V * (w.shape[0] / r) ** .5
  w.add_(u, alpha=-lr)
