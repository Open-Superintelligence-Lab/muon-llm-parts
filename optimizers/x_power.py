"""Muon's step with singular values raised to a power p instead of to 0, at Muon's size. p=0 is exact orthogonalization, p above 0 keeps some of the gradient's ranking of directions, and p below 0 favours weak directions over strong ones."""
import torch


def step(w, g, st, lr, k):
    beta, p = k.get("beta", .95), k.get("p", 0.); h = int(k.get("h", 1)); n = w.shape[0] // h
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(beta).add_(g); d = g + beta * st["b"]
    for j in range(h):
        U, S, V = torch.linalg.svd(d[j * n:(j + 1) * n], full_matrices=False); s = S / (S[0] + 1e-12); f = s.clamp(min=1e-3) ** p; f = f * (len(S) / (f * f).sum()) ** .5
        w[j * n:(j + 1) * n].add_((U * f) @ V * max(1.0, n / w.shape[1]) ** .5, alpha=-lr)
