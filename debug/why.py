"""Why did a design not beat the baseline? Trains the tuned muon-split arm and, at every step, also computes what each design would have done to query and key from the same weights and gradient. Prints the cosine with the real step. Training itself uses only muon-split.

python3 debug/why.py [--steps 120]
"""
import argparse, json, os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import torch
import arms, env, rules, sandbox

p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=120); o = p.parse_args()
NAMES = ["x_head_zigzag", "x_head_partial", "x_head_trust", "x_whole_head", "x_muon_lion", "x_head_apart", "x_ada_muon", "x_ortho_first", "x_partial", "x_two_speed"]
F = {n: sandbox.load(n) for n in NAMES}; F["muon"] = rules.RULES["muon"]; S = {n: {} for n in F}; C = {n: [0., 0., 0] for n in F}; real = rules.RULES["muonsplit"]; seen = [0]


def spy(w, g, st, lr, k):
    k2 = {x: v for x, v in k.items() if x != "params"}; old = w.clone(); out = {}
    for n, f in F.items():
        c = old.clone(); f(c, g, S[n].setdefault(id(w), {}), lr, k2 if n == "muon" else k); out[n] = c - old
    real(w, g, st, lr, k); d = w - old; seen[0] += 1
    if seen[0] > o.steps:  # the second half of training, 2 weights per layer, once the momentum has settled
        for n, x in out.items(): C[n][0] += float((x * d).sum() / (x.norm() * d.norm() + 1e-12)); C[n][1] += float(x.norm() / (d.norm() + 1e-12)); C[n][2] += 1


rules.RULES["muonsplit"] = spy
ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; spec = arms.parse([r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == "muon-split" and r["tier"] == "5s"][-1]["spec"])
a = env.parser().parse_args(["--spec", arms.show(spec), "--tier", "5s", "--steps", str(o.steps), "--results", os.path.join(tempfile.mkdtemp(), "r.jsonl")]); out = a.results
env.train(a, spec, None, out)
for n in sorted(C, key=lambda n: -C[n][0]): print(f"{n}: cosine with the real step {C[n][0] / max(1, C[n][2]):.3f}, size {C[n][1] / max(1, C[n][2]):.2f}x")
