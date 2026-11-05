"""Muon with one memory for the strong directions and another for the weak ones.

fast is the strong directions' momentum, slow is the weak directions'. q is the share of directions counted as strong. Equal momentums give exact Muon by SVD. ns=1 joins the strong part of the fast memory with the weak part of the slow memory and orthogonalizes by Newton-Schulz, and then equal momentums give Muon itself.
The strong directions come from the fast memory. The slow memory then has those directions removed and the rest is orthogonalized, so the step stays equal in every direction.
vote=1 orthogonalizes each gradient before it enters the slow memory, so loud gradients get no extra say in the weak directions.
"""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def orth(d):
    U, S, V = torch.linalg.svd(d, full_matrices=False)
    return U, V


def step(w, g, st, lr, k):
    beta = k.get("beta", .95); h = int(k.get("h", 1)); n = w.shape[0] // h; fast, slow = k.get("fast", beta), k.get("slow", beta); share = k.get("q", .25); vote = k.get("vote", 0)
    if "f" not in st: st["f"] = torch.zeros_like(w); st["s"] = torch.zeros_like(w)
    st["f"].mul_(fast).add_(g); df = g + fast * st["f"]
    if vote:
        o = torch.zeros_like(w)
        for j in range(h):
            a, c = j * n, (j + 1) * n; U, V = orth(g[a:c]); o[a:c] = U @ V
        st["s"].mul_(slow).add_(o); ds = o + slow * st["s"]
    else: st["s"].mul_(slow).add_(g); ds = g + slow * st["s"]
    for j in range(h):
        a, c = j * n, (j + 1) * n; U, V = orth(df[a:c]); r = len(U[0]); q = max(1, int(r * share)); Ut, Vt = U[:, :q], V[:q]
        e = ds[a:c]; e = e - Ut @ (Ut.T @ e); e = e - (e @ Vt.T) @ Vt
        if k.get("ns", 0):
            U2, S2, V2 = torch.linalg.svd(df[a:c], full_matrices=False); keep = (U2[:, :q] * S2[:q]) @ V2[:q]; rest = df[a:c] - keep; x = ns(keep + e * (rest.norm() / (e.norm() + 1e-12)))
        else:
            P, R = orth(e); x = Ut @ Vt + P[:, :r - q] @ R[:r - q]
        w[a:c].add_(x * max(1.0, n / w.shape[1]) ** .5, alpha=-lr)
