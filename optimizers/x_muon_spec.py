"""Muon's singular directions with a different size for each one. p sets size = singular value ** p (p=0 is exact Muon, p=1 keeps the gradient's own shape). rev=1 gives the strongest direction the weakest one's size and so on. rand=1 draws sizes from a chaotic number sequence that leaves the training's random numbers alone. Every step is scaled to Muon's size, so only the shape changes."""
import torch


def step(w, g, st, lr, k):
    beta = k.get("beta", .95)
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(beta).add_(g); d = g + beta * st["b"]
    U, S, V = torch.linalg.svd(d, full_matrices=False); s = S / (S[0] + 1e-12)
    if k.get("rand", 0):
        if "x" not in st: st["x"] = .1 + .8 * (torch.arange(len(S), dtype=d.dtype) + .5) / len(S)
        for i in range(3): st["x"].copy_(3.99 * st["x"] * (1 - st["x"]))
        f = st["x"] + .01
    else:
        f = (s + 1e-12) ** k.get("p", 0)
        if k.get("rev", 0): f = f.flip(0)
    f = f * len(S) ** .5 / (f.norm() + 1e-12)
    w.add_((U * f) @ V * max(1.0, w.shape[0] / w.shape[1]) ** .5, alpha=-lr)
