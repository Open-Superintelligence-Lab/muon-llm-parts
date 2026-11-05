"""MuonM's update (2608.28442) without its tangent projection: Muon's step on the fast momentum, plus chi times a slow momentum with the strong directions removed. chi=0 is Muon. slow is the slow momentum's decay, q the share of directions treated as strong. sphere=1 holds each row's norm at its first value, as MuonS and MuonM do."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    beta, slow, chi = k.get("beta", .95), k.get("slow", .99), k.get("chi", 0.); h = int(k.get("h", 1)); n = w.shape[0] // h
    if "f" not in st: st.update(f=torch.zeros_like(w), s=torch.zeros_like(w), r=w.norm(dim=1, keepdim=True))
    st["f"].mul_(beta).add_(g); st["s"].mul_(slow).add_(g, alpha=1 - slow); d = g + beta * st["f"]; size = max(1.0, n / w.shape[1]) ** .5
    for j in range(h):
        a, c = j * n, (j + 1) * n; x = ns(d[a:c])
        if chi:
            e = st["s"][a:c]; e = e * (x.norm(dim=1, keepdim=True) / (e.norm(dim=1, keepdim=True) + 1e-12)); U, S, V = torch.linalg.svd(d[a:c], full_matrices=False); q = max(1, int(len(S) * k.get("q", .25))); U, V = U[:, :q], V[:q]
            e = e - U @ (U.T @ e); e = e - (e @ V.T) @ V; x = x + chi * e
        w[a:c].add_(x * size, alpha=-lr)
    if k.get("sphere"): w.mul_(st["r"] / (w.norm(dim=1, keepdim=True) + 1e-12))
