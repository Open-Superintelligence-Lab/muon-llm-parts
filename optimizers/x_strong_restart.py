"""Muon that forgets a strong direction when it overshoots. If the new gradient opposes a strong direction of the last step, that direction is removed from the momentum, so only directions that overshot lose their memory. q is the share of directions watched. Each row group keeps its own record."""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    beta = k.get("beta", .95); h = int(k.get("h", 1)); n = w.shape[0] // h; share = k.get("q", .25); keep = k.get("keep", 0)
    if "b" not in st: st["b"] = torch.zeros_like(w)
    st["b"].mul_(beta).add_(g)
    for j in range(h):
        a, c = j * n, (j + 1) * n
        if f"u{j}" in st:
            U, V = st[f"u{j}"], st[f"v{j}"]; v = (U.T @ g[a:c] @ V).diagonal(); m = (U.T @ st["b"][a:c] @ V).diagonal(); bad = (v < 0).float() * (1 - keep)
            st["b"][a:c].sub_((U * (bad * m)) @ V.T)
    d = g + beta * st["b"]
    for j in range(h):
        a, c = j * n, (j + 1) * n; U, S, Vh = torch.linalg.svd(d[a:c], full_matrices=False); q = max(1, int(len(S) * share))
        st[f"u{j}"] = U[:, :q].clone(); st[f"v{j}"] = Vh[:q].T.clone()
        w[a:c].add_(ns(d[a:c]) * max(1.0, n / w.shape[1]) ** .5, alpha=-lr)
