"""Random stages, all 0 is Muon. See debug/random_opt.py."""
import math
import torch


def ns(d, n=5):
  X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
  if tall: X = X.T
  for i in range(n):
    A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
  return X.T if tall else X


def unit(z, s): return z / (z.norm() + 1e-7) * s


def rows(x): return x.norm(dim=1, keepdim=True) + 1e-8


def cols(x): return x.norm(dim=0, keepdim=True) + 1e-8


def rms(x): return x.pow(2).mean().sqrt() + 1e-12


def spec(x, f):
  U, S, V = torch.linalg.svd(x / (x.norm() + 1e-7), full_matrices=False); return (U * f(S.clamp_min(1e-6))) @ V


def before(g, st, n, p, t):
  if n == 1: return g - g.mean(1, keepdim=True)
  if n == 2: return g - g.mean(0, keepdim=True)
  if n == 3: return g / (g.norm() + 1e-12)
  if n == 4: return g / rows(g)
  if n == 5: return g / cols(g)
  if n == 6: return (1 - p) * g / rms(g) + p * g.sign()
  if n == 7:
    st.setdefault("v", torch.zeros_like(g)).mul_(.95).addcmul_(g, g, value=.05); return g / ((st["v"] / (1 - .95 ** t)).sqrt() + 1e-8) ** p
  if n == 8:
    c = (1 + 3 * p) * rms(g); return g.clamp(-c, c)
  if n == 9:
    r = rows(g); return g * ((1 + p) * r.median() / r).clamp_max(1)
  if n == 10:
    o = st["og"] if "og" in st else g; st["og"] = g.clone(); return g + p * (g - o)
  return g


def memory(g, st, n, p, beta):
  if n == 0:
    st.setdefault("b", torch.zeros_like(g)).mul_(beta).add_(g); return g + beta * st["b"]
  if n in (2, 9): return g
  m = st.setdefault("m", torch.zeros_like(g))
  if n == 6 and (g * m).sum() < 0: m.mul_(0)
  if n == 7: m.mul_(torch.where(g * m < 0, beta * p, beta)).add_(g, alpha=1 - beta)
  else:
    if n == 8: m.mul_(((g * m).sum(1, keepdim=True) >= 0).float())
    m.mul_(beta).add_(g, alpha=1 - beta)
  if n == 3:
    st.setdefault("f", torch.zeros_like(g)).mul_(.5).add_(g, alpha=.5); return (1 - p) * m + p * st["f"]
  if n == 4: return (1 - p) * g + p * m
  if n == 5:
    o = st["og2"] if "og2" in st else g; st["og2"] = g.clone(); return m + p * (g - o)
  return m


def shaped(x, n, p, h, s):
  if n == 0: return ns(x)
  if n == 1: return x
  if n == 2: return spec(x, lambda S: S ** p)
  if n == 3: return x / rows(x)
  if n == 4: return x / cols(x)
  if n == 5:
    y = x / rows(x); y = y / cols(y); return y / rows(y)
  if n == 6: return x.sign()
  if n == 7: return torch.tanh(x / ((.2 + 2 * p) * rms(x)))
  if n == 8:
    tall = x.shape[0] >= x.shape[1]; Q, R = torch.linalg.qr(x if tall else x.T); Q = Q * R.diagonal().sign(); return Q if tall else Q.T
  if n == 9:
    a = x.shape[0] // h; return torch.cat([ns(x[i:i + a]) for i in range(0, x.shape[0], a)], 0)
  if n == 10:
    a = x.shape[1] // h; return torch.cat([ns(x[:, i:i + a]) for i in range(0, x.shape[1], a)], 1)
  if n == 11: return spec(x, lambda S: S.clamp_max(torch.quantile(S, p)))
  if n == 12: return spec(x, lambda S: (S >= torch.quantile(S, p)).float())
  if n == 13: return spec(x, lambda S: torch.cat([S[:1] * 0, S[1:] ** p]))
  if n == 14: return spec(x, lambda S: torch.log1p(S / (S.median() + 1e-9)))
  if n == 15: return (1 - p) * unit(ns(x), s) + p * unit(x.sign(), s)
  if n == 16: return (1 - p) * unit(ns(x), s) + p * unit(x, s)
  if n == 17: return (1 - p) * unit(ns(x), s) + p * unit(x / (x.abs() + rms(x)), s)
  if n == 18: return ns(x, 1 + int(3 * p))
  if n == 19: return x.sign() * (x.abs().reshape(-1).argsort().argsort().float().reshape(x.shape) / x.numel()) ** p
  if n == 20: return x * (x.abs() >= torch.quantile(x.abs().reshape(-1)[:9000], p))
  if n == 21:
    z = ns(x); return z / rows(z) * rows(x) ** p
  if n == 22: return ns(x) * (x.abs() / rms(x)) ** p
  if n == 23: return ns(x / rows(x))
  if n == 24: return ns(x / cols(x))
  return x


def step(w, g, st, lr, k):
  pre, mom, sh, si, po = [int(k.get(x, 0)) for x in ("pre", "mom", "shape", "size", "post")]
  p1, p2, p3, p4, p5 = [k.get(x, .5) for x in ("p1", "p2", "p3", "p4", "p5")]; beta = k.get("beta", .95); h = int(k.get("h", 4)); t = st["t"] = st.get("t", 0) + 1; f = k.get("t", False); m0 = w.shape[0]; s = m0 ** .5
  if w.ndim != 2: return
  g = before(g, st, pre, p1, t); d = torch.nan_to_num(memory(g, st, mom, p2, beta)); x = d.T if f else d; z = shaped(x, sh, p3, h, s)
  if mom == 9: m = st.setdefault("m", torch.zeros_like(z)); m.mul_(beta).add_(z, alpha=1 - beta); z = m
  if si == 4: r = (w.T if f else w).norm(dim=1, keepdim=True); z = z * (r / (r.mean() + 1e-9)) ** p4
  z = z.T if f else z
  if si == 8: y = z.T if f else z; b = w.T if f else w; y = y - (y * b).sum(1, keepdim=True) / (b * b).sum(1, keepdim=True).clamp_min(1e-9) * b; z = y.T if f else y
  if si == 7: z = z - (z * w).sum() / ((w * w).sum() + 1e-12) * w
  z = z * max(1.0, m0 / w.shape[1]) ** .5 if sh == 0 and mom != 9 and si in (0, 1, 2, 3, 5, 6, 9) else unit(z, s)
  a = lr
  if si == 1: st["e"] = .9 * st.get("e", float(g.norm())) + .1 * float(g.norm()); a = a * (float(g.norm()) / (st["e"] + 1e-12)) ** p4
  if si == 2: a = a * (1 + p4 * float((g * d).sum() / (g.norm() * d.norm() + 1e-12)))
  if si == 3: st.setdefault("w0", float(w.norm())); a = a * (float(w.norm()) / st["w0"]) ** p4
  if si == 5: a = a / (1 + t / (50 + 400 * p4))
  if si == 6: a = a * (1 + t / (50 + 400 * p4))
  if si == 9: a = a * (1 + p4 * math.sin(6.2832 * t / 100))
  w.add_(z, alpha=-a)
  if po == 1:
    st.setdefault("slow", w.clone())
    if t % (5 + int(20 * p5)) == 0: w.add_(st["slow"] - w, alpha=.5); st["slow"].copy_(w)
  if po == 2: st.setdefault("r0", float(w.norm())); w.mul_(st["r0"] / (w.norm() + 1e-12))
  if po == 3: w.mul_(1 - a * p5)
  if po == 4:
    tall = w.shape[0] >= w.shape[1]; C = w.T @ w if tall else w @ w.T; n = C.shape[0]; P = w @ C if tall else C @ w; w.add_(P * n / (C.diagonal().sum() + 1e-9) - w, alpha=-a * p5)
  if po == 5:
    b = w.T if f else w; st.setdefault("rn", b.norm(dim=1, keepdim=True).clone()); b.mul_(st["rn"] / (b.norm(dim=1, keepdim=True) + 1e-12))
  if po == 6: w.add_(w.mean(0, keepdim=True), alpha=-a * p5)
