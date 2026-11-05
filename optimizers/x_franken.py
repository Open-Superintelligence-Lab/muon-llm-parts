"""Muon with pieces from other optimizers, each switched by a knob and off at 0. With every piece off this is Muon (h=1) or per-head Muon (h=4).

center: gradient centralization, removes each row's mean from the gradient.
gnorm: normalized SGD, every gradient enters the momentum at the same size.
nes=0: heavy-ball momentum in place of Nesterov.
ada: Adam's division by gradient size before orthogonalizing, to this power.
graft: optimizer grafting, Muon's direction with Adam's step size, to this power.
lion: this share of the step is Lion's sign step, scaled to the same size.
nsn: Newton-Schulz steps, fewer steps leave the singular values less equal.
look: Lookahead, every this many steps the weight moves halfway back to its slow copy.
warm: the lr grows over this many steps.
fade: the lr halves every this many steps.
wd: weight decay on top of the step.
"""
import torch


def ns(d, steps=5):
    X = d / (d.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for i in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def step(w, g, st, lr, k):
    beta = k.get("beta", .95); h = int(k.get("h", 1)); n = w.shape[0] // h; adam = k.get("ada", 0) or k.get("graft", 0)
    if "b" not in st:
        st["b"] = torch.zeros_like(w); st["t"] = 0
        if adam: st["v"] = torch.zeros_like(w)
        if k.get("look", 0): st["slow"] = w.clone()
    st["t"] += 1; t = st["t"]
    if k.get("center", 0): g = g - g.mean(dim=1, keepdim=True)
    if k.get("gnorm", 0): g = g / (g.norm() + 1e-12)
    st["b"].mul_(beta).add_(g); d = g + beta * st["b"] if k.get("nes", 1) else st["b"]
    if adam: st["v"].mul_(.95).addcmul_(g, g, value=.05); r = (st["v"] / (1 - .95 ** t)).sqrt() + 1e-8
    if k.get("ada", 0): d = d / r ** k["ada"]
    size = lr
    if k.get("warm", 0): size = size * min(1, t / k["warm"])
    if k.get("fade", 0): size = size * .5 ** (t / k["fade"])
    if k.get("wd", 0): w.mul_(1 - size * k["wd"])
    for j in range(h):
        a, c = j * n, (j + 1) * n; x = ns(d[a:c], int(k.get("nsn", 5))) * max(1.0, n / w.shape[1]) ** .5
        if k.get("lion", 0): x = (1 - k["lion"]) * x + k["lion"] * st["b"][a:c].sign() * (x.norm() / x.numel() ** .5)
        if k.get("graft", 0): x = x * ((st["b"][a:c] * (1 - beta) / (1 - beta ** t) / r[a:c]).pow(2).mean().sqrt() ** k["graft"])
        w[a:c].add_(x, alpha=-size)
    if k.get("look", 0) and t % int(k["look"]) == 0: w.add_(st["slow"] - w, alpha=.5); st["slow"].copy_(w)
