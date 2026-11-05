"""How does the exponent c split a fixed-size update between top and small singular directions? Runs one training run and, every --every steps, takes the singular values S of each hidden matrix's Nesterov momentum. For each c, the update U S^(1-2c) V^T resized to Frobenius size sqrt(rows) has singular values a_i; prints, per part, the mean top value a_1 and the share of |a|^2 in the bottom half of directions, both relative to c=.5 (flat spectrum), and the RMS step in each quarter of directions relative to the flat update. If the exponent mainly sets the top step, the best-lr ratio between two c should match the ratio of their a_1.

python3 debug/top_step.py "SPEC" [--tier 5sx4] [--world 0] [--every 50]
"""
import argparse, collections, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "debug"))
import map_parts  # noqa: F401
import numpy as np, torch
import arms, env, sandbox

p = argparse.ArgumentParser(); p.add_argument("spec"); p.add_argument("--tier", default="5sx4"); p.add_argument("--world", type=int, default=0); p.add_argument("--every", type=int, default=50); a0 = p.parse_args()
NAME = {}; COUNT = collections.Counter(); SPEC = collections.defaultdict(list); CS = (.3, .4, .5, .6, .7, .8)
real_groups = arms.groups


def groups(spec, named):
    for n, w in named: NAME[w.data_ptr()] = arms.module_of(n, w)
    return real_groups(spec, named)


arms.groups = groups; real_wrap = sandbox.wrap


def wrap(fn):
    inner = real_wrap(fn)

    def rule(w, g, st, lr, k):
        key = w.data_ptr(); t = COUNT[key] = COUNT[key] + 1; G = g.detach().float()
        inner(w, g, st, lr, k)
        if t % a0.every == 0 and "b" in st and G.ndim == 2:
            d = G + k.get("beta", .95) * st["b"].float(); SPEC[NAME.get(key, "?")].append((t, torch.linalg.svdvals(d / (d.norm() + 1e-7)).clamp_min(1e-6).numpy(), w.shape[0]))
    return rule


sandbox.wrap = wrap
a = env.parser().parse_args(["--spec", a0.spec, "--tier", a0.tier, "--world", str(a0.world), "--results", os.path.join(os.environ.get("TMPDIR", "/tmp"), "top_step.jsonl"), "--force"])
env.train(a, env.full_spec(a))
part_of = dict(q="qk", k="qk", v="vo", o="vo", gate="ffn", up="ffn", down="ffn")
R = collections.defaultdict(lambda: collections.defaultdict(list))
for m, rows in SPEC.items():
    for t, S, n in rows:
        for c in CS:
            f = S ** (1 - 2 * c); f = f * (n ** .5 / np.linalg.norm(f)); h = len(f) // 2
            flat = (n / len(f)) ** .5; R[part_of.get(m, m)][c].append((f[0], (f[h:] ** 2).sum() / (f ** 2).sum(), *[float(np.sqrt((q ** 2).mean()) / flat) for q in np.array_split(f, 4)]))
for part, d in sorted(R.items()):
    ref = np.mean([x[0] for x in d[.5]])
    print(f"{part}: c, top singular value of the update relative to c=.5, bottom-half share of |update|^2")
    for c in CS: v = np.array(d[c]); print(f"  c={c}: top x{v[:, 0].mean() / ref:.2f}, bottom half {v[:, 1].mean():.3f}, RMS step per quarter vs flat " + " ".join(f"{x:.2f}" for x in v[:, 2:].mean(0)), flush=True)
