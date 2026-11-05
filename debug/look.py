"""Looks inside one short run of a tuned arm. Training is unchanged; every step also records, from the next step's gradient, what the last step did.

For each part: its share of the first-order loss drop per phase, and step size over weight size. For Muon's matrices: the step's singular directions split into 4 groups from strongest to weakest. "still going" is u'gv / |g| with the next gradient: above zero means the loss still falls along that direction, below zero means the step overshot.

python3 debug/look.py [--arm muon-split] [--world 0] [--steps 250]
"""
import argparse, json, os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np, torch
import arms, env, rules

p = argparse.ArgumentParser(); p.add_argument("--arm", default="muon-split"); p.add_argument("--world", type=int, default=0); p.add_argument("--steps", type=int, default=250); o = p.parse_args()
ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; spec = arms.parse([r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == o.arm and r["tier"] == "5s"][-1]["spec"])
NAME = {(g["rule"], round(g["lr"], 6)): m for m, g in spec.items() if m not in ("router", "exp", "ed")}; PREV = {}; COUNT = {}; DROP = {}; SIZE = {}; DIR = {}; RANK = {}; PHASES = ((0, 30), (30, 100), (100, 10 ** 9))


def phase(t): return next(i for i, (a, b) in enumerate(PHASES) if a <= t < b)


def spy(rule, name):
    def f(w, g, st, lr, k):
        m = NAME[(name, round(k["lr"], 6))]; i = id(w); t = COUNT.get(i, 0); COUNT[i] = t + 1; ph = phase(t)
        if i in PREV:
            dw, svds = PREV[i]; DROP[(m, ph)] = DROP.get((m, ph), 0) + float(-(g * dw).sum())
            for rows, U, S, V in svds:
                x = g[rows]; v = (U.T @ x @ V).diagonal() / (x.norm() + 1e-12); n = len(v)
                for b in range(4):
                    DIR.setdefault((m, ph, b), []).append(float(v[b * n // 4:max((b + 1) * n // 4, b * n // 4 + 1)].mean()))
        old = w.detach().clone(); rule(w, g, st, lr, k); dw = w.detach() - old; SIZE.setdefault((m, ph), []).append(float(dw.norm() / w.norm())); svds = []
        if "muon" in name:
            d = g + k.get("beta", .95) * st["b"]; n = w.shape[0] // int(k.get("h", 4)) if name == "muonsplit" else w.shape[0]
            for a in range(0, w.shape[0], n):
                U, S, Vh = torch.linalg.svd(d[a:a + n], full_matrices=False); svds.append((slice(a, a + n), U, S, Vh.T)); RANK.setdefault((m, ph), []).append(float((S ** 2).sum() / S[0] ** 2) / len(S))
        PREV[i] = (dw, svds)
    return f


for r in {g["rule"] for g in spec.values()}: rules.RULES[r] = spy(rules.RULES[r], r)
a = env.parser().parse_args(["--spec", arms.show(spec), "--tier", "5s", "--world", str(o.world), "--steps", str(o.steps), "--results", os.path.join(tempfile.mkdtemp(), "r.jsonl")]); env.train(a, spec, None, a.results)
parts = sorted({m for m, _ in DROP}, key=lambda m: -sum(DROP.get((m, q), 0) for q in range(3))); print("phases: steps 0-30, 30-100, 100-end")
for q in range(3):
    tot = sum(DROP.get((m, q), 0) for m in parts); print(f"phase {q} total drop {tot:.3f}: " + " ".join(f"{m} {DROP.get((m, q), 0) / tot:+.0%}" for m in parts))
for m in parts: print(m, "step/weight " + " ".join(f"{np.mean(SIZE[(m, q)]):.4f}" for q in range(3)), ("stable rank share " + " ".join(f"{np.mean(RANK[(m, q)]):.2f}" for q in range(3))) if (m, 0) in RANK else "")
print("still going, strongest to weakest directions:")
for m in parts:
    for q in range(3):
        if (m, q, 0) in DIR: print(f"{m} phase {q}: " + " ".join(f"{np.mean(DIR[(m, q, b)]):+.4f}" for b in range(4)))
