"""Muon with a learned spectrum exponent per matrix. Update U S^(1-2c) V^T of the Nesterov momentum, resized to Muon's size; c=.5 is the exact polar direction. After each step, c moves by hypergradient descent: the next gradient's cosine with d(update)/dc says whether a larger c would have lowered the loss. Knobs: c0 start, eta exponent step, beta momentum, lo/hi bounds, rho horizon: d(update)/dc is summed over past steps with decay rho, so the cosine scores the exponent by its effect over about 1/(1-rho) steps, not only the last one (rho=0)."""
import torch


def step(w, g, st, lr, k):
  if w.ndim != 2: return
  beta = k.get("beta", .95); eta = k.get("eta", .01); lo = k.get("lo", 0.0); hi = k.get("hi", 1.0); rho = k.get("rho", 0.0)
  c = st.get("c", k.get("c0", .5))
  if "h" in st:
    h = st["h"]; cos = float((g * h).sum() / (g.norm() * h.norm() + 1e-12)); c = min(hi, max(lo, c + eta * cos))
  st["c"] = c
  b = st.setdefault("b", torch.zeros_like(g)); b.mul_(beta).add_(g); d = torch.nan_to_num(g + beta * b)
  U, S, V = torch.linalg.svd(d / (d.norm() + 1e-7), full_matrices=False); S = S.clamp_min(1e-6)
  f = S ** (1 - 2 * c); u = (U * f) @ V
  r = w.shape[0] ** .5; un = u.norm() + 1e-12; u = u * (r / un)
  h = (U * (-2 * torch.log(S) * f)) @ V * (r / un); h = h - (h * u).sum() / (u * u).sum() * u
  if "h" in st: h = st["h"].mul_(rho).add_(h)
  st["h"] = h
  w.add_(u, alpha=-lr)
