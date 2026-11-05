"""Why does each group want its step shape? Diagnostic only: one training run of the tuned Muon baseline, no scores. At each measured step it takes the singular directions of Muon's momentum and the gradient of the next batch, which training has not seen, at the same weights. For a step shape f it reports the cosine between that step and the unseen gradient: higher means the step lowers unseen loss faster per unit of step. flat is Muon, grad keeps the singular values as sizes, top uses only the strongest quarter of directions, weak only the rest.

python3 debug/spectrum.py 5s results/add/w48_out.jsonl results/spec/probe_w48.json
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_parts as mp
import arms, env, rules
import torch
import torch.nn.functional as F

tier, base_file, out = sys.argv[1:4]; b = arms.parse([json.loads(l) for l in open(base_file) if "gaps" in l][-1]["base"]); a = env.parser().parse_args(["--tier", tier]); a.world = int(sys.argv[4]) if len(sys.argv) > 4 else 0; a.evals = "tune"; a.threads = int(os.environ.get("OMP_NUM_THREADS", 2))
c = dict(env.TIERS[tier]); steps = c["steps"]; every = max(1, steps // 100); W = env.World(c, steps, a.world); X = torch.as_tensor(W.look[W.rows]); rows = []; count = [0]; plain = rules.ModuleOpt.step; held = []; LM = env.LM


def hold(*x, **k):
    held.append(LM(*x, **k)); return held[-1]


def cos(f, x, gn): return float((f * x).sum() / (f.norm() * gn + 1e-12))


def probed(s, sc=1.0, measure=False):
    t = count[0]; count[0] += 1
    if t % every == 0 and t + 1 < steps:
        ps = [(m, g, w) for m, g in s.groups.items() if g["rule"] == "muon" for w in g["params"] if w.grad is not None and "b" in s.state[id(w)]]
        if ps:
            model = held[-1]; x = X[((t + 1) * c["batch"] + torch.arange(c["batch"]).numpy()) % len(X)]
            loss = F.cross_entropy(model(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten(), ignore_index=W.unk); Gs = torch.autograd.grad(loss, [w for _, _, w in ps])
            for (m, g, w), G in zip(ps, Gs):
                beta = g.get("beta", .95); d = w.grad + beta * (beta * s.state[id(w)]["b"] + w.grad); U, S, Vh = torch.linalg.svd(d.detach(), full_matrices=False); e = ((U.T @ G) * Vh).sum(1); q = max(1, len(S) // 4); gn = float(G.norm()); top = torch.zeros_like(S); top[:q] = 1; one = torch.ones_like(S)
                rows.append(dict(t=t, m=m, flat=cos(one, e, gn), grad=cos(S, e, gn), top=cos(top, e, gn), weak=cos(one - top, e, gn), energy=float(S[:q].pow(2).sum() / S.pow(2).sum())))
    plain(s, sc, measure)


env.LM = hold; rules.ModuleOpt.step = probed; env.train(a, b, W=W, out=os.path.join(tempfile.mkdtemp(), "x.jsonl")); res = {}
for name, parts in dict(vo=("v", "o"), qk=("q", "k"), ffn=("gate", "up", "down")).items():
    for label, lo, hi in (("first 20%", 0, steps / 5), ("middle", steps / 5, steps * .7), ("last 30%", steps * .7, steps), ("whole run", 0, steps)):
        x = [e for e in rows if e["m"] in parts and lo <= e["t"] < hi]
        if x: res[f"{name} {label}"] = dict({k: round(sum(e[k] for e in x) / len(x), 4) for k in ("flat", "grad", "top", "weak", "energy")}, samples=len(x)); print(name, label, res[f"{name} {label}"], flush=True)
os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(dict(tier=tier, world=a.world, base=arms.show(b), every=every, rows=res), open(out, "w"), indent=0)
