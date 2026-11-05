"""Muon's singular directions with one change of step shape during training. size = singular value ** p, p1 and lr factor f1 before step at, p2 and f2 from then on. p=0 is exact Muon, p=1 keeps the gradient's own shape."""
import torch


def step(w, g, st, lr, k):
    beta = k.get("beta", .95)
    if "b" not in st: st["b"] = torch.zeros_like(w); st["n"] = 0
    st["n"] += 1; late = st["n"] > k.get("at", 0)
    st["b"].mul_(beta).add_(g); d = g + beta * st["b"]
    U, S, V = torch.linalg.svd(d, full_matrices=False); s = S / (S[0] + 1e-12)
    f = (s + 1e-12) ** k.get("p2" if late else "p1", 0)
    f = f * len(S) ** .5 / (f.norm() + 1e-12)
    w.add_((U * f) @ V * max(1.0, w.shape[0] / w.shape[1]) ** .5, alpha=-lr * k.get("f2" if late else "f1", 1))
