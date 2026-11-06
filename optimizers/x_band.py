"""Polar update with a set step per quarter of singular directions. Takes the exact SVD of the Nesterov momentum d = U S V^T, sorts directions by singular value, and steps U diag(m) V^T * sqrt(rows / rank), where m is t1 for the top quarter, t2, t3 and t4 for the next ones. All t equal to 1 is exact polar at Muon's update size. Used to ask which quarter of the spectrum carries the exponent's gain. flip4=1 gives each bottom-quarter direction a random sign every step, which keeps its size and removes its direction. With at=N, steps after the N-th use u1..u4 in place of t1..t4 (each defaulting to its t)."""
import torch


def step(w, g, st, lr, k):
  if w.ndim != 2: return
  beta = k.get("beta", .95)
  b = st.setdefault("b", torch.zeros_like(g)); b.mul_(beta).add_(g); d = torch.nan_to_num(g + beta * b)
  st["n"] = st.get("n", 0) + 1; late = "at" in k and st["n"] > k["at"]
  U, S, V = torch.linalg.svd(d, full_matrices=False); r = S.shape[0]
  m = torch.ones_like(S); q = torch.arange(r, device=S.device) * 4 // r
  for i in range(4): t = k.get(f"t{i + 1}", 1.0); m[q == i] = k.get(f"u{i + 1}", t) if late else t
  if k.get("flip4", 0): m = torch.where((q == 3) & (torch.rand_like(S) < .5), -m, m)
  u = (U * m) @ V * (w.shape[0] / r) ** .5
  w.add_(u, alpha=-lr)
