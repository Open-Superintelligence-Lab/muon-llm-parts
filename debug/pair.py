"""Compares changes to a full spec, part by part, on chosen worlds, on tune rows. A change is part=rule:lr[:knob=value...] with + joining changes, and a part's rule name may be left empty to keep it. Positive means better than the base.

python3 debug/pair.py "<spec>" qk=muon:.21:beta=.8 "gu=muon:.03+vo=x_blocks:.05:rows=4" --worlds 5,6,7,8,9 --results path
"""
import argparse, json, os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
import arms, env, runq
env.TIERS["long"] = dict(width=32, layers=2, heads=4, V=1000, batch=8, L=48, steps=1000, val_rows=3200, marks=5, bench=60)  # about 10 s; workers import this file again, so they get it too


FINE = dict(q="qk", k="qk", v="vo", o="vo", gate="gu", up="gu")
if os.environ.get("FINE"):  # q, k, v, o, gate and up get their own settings, for this run only; arms.py stays as it is, and workers inherit this
    old = arms.module_of
    arms.module_of = lambda n, w: (lambda m: m.group(1) if m else old(n, w))(re.search(r"\.(q|k|v|o|gate|up)(\.\d+)?\.weight$", n))


def split(b):
    if not os.environ.get("FINE"): return b
    out = {m: g for m, g in b.items() if m not in ("qk", "vo", "gu")}
    for m, p in FINE.items(): out[m] = dict(b[p])
    return out


def change(base, text):
    s = {m: dict(g) for m, g in base.items()}
    for one in text.split("+"):
        m, _, rest = one.partition("="); bits = rest.split(":"); g = dict(lr=float(bits[1])); old = s[m]; g["rule"] = bits[0] or old["rule"]
        if not bits[0]: g.update({k: v for k, v in old.items() if k not in ("rule", "lr")})
        for b in bits[2:]: g[b.split("=")[0]] = float(b.split("=")[1])
        s[m] = g
    return s


def main():
    p = argparse.ArgumentParser(); p.add_argument("base"); p.add_argument("changes", nargs="*"); p.add_argument("--worlds", default="5,6,7,8,9"); p.add_argument("--results", required=True); p.add_argument("--seconds", type=float, default=500); p.add_argument("--tier", default="5s"); a = p.parse_args()
    base = split(arms.parse(a.base)); W = [int(w) for w in a.worlds.split(",")]; specs = {c: change(base, c) for c in a.changes}
    job = lambda s, w: env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--world", str(w), "--evals", "tune", "--tag", "pair", "--results", a.results])
    left = runq.run([job(s, w) for s in [base] + list(specs.values()) for w in W], a.seconds)
    if runq.EX: runq.EX.shutdown()
    if left: return print(left, "runs left: rerun the same command")
    rows = {}
    for l in open(a.results):
        r = json.loads(l); rows[(r["spec"], r["world"])] = r["scores"]["text.tune.loss"] if r["finite"] and not r["opt"]["unstable"] else 99.
    val = lambda s: np.array([rows[(arms.show(s), w)] for w in W]); b = val(base); print(f"base {b.mean():.4f}", flush=True)
    for c, s in specs.items():
        g = b - val(s); print(f"{c}: {g.mean():+.4f} error {g.std(ddof=1) / len(W) ** .5:.4f} wins {(g > 0).sum()}/{len(W)} {np.round(g, 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
