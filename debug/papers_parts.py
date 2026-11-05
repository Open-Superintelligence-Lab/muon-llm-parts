"""Kaon, Freon (2605.11181) and Dion3 (microsoft/dion) on one part at a time, then on every hidden matrix. Freon is x_rand's spectrum power at 1 - 2c, resized to Muon's Frobenius norm. Tunes lr then momentum on worlds 0,1 and scores on worlds 5-9, tune rows.

python3 debug/papers_parts.py 5s results/add/w48_out.jsonl --results results/papers/w48.jsonl --out results/papers/w48_out.json [--parts qk] [--only kaon] [--sweep] [--par 9] [--hours H]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

PARTS = dict(qk=("q", "k"), vo=("v", "o"), ffn=("gate", "up", "down"), all=("q", "k", "v", "o", "gate", "up", "down"))
ARMS = {"kaon": dict(rule="x_kaon"), "freon c=1/4": dict(rule="x_rand", shape=2, p3=.5), "freon c=2/3": dict(rule="x_rand", shape=2, p3=-.3333), "freon c=3/4": dict(rule="x_rand", shape=2, p3=-.5), "freon c=1": dict(rule="x_rand", shape=2, p3=-1), "dion3 f=.25": dict(rule="x_dion3", f=.25), "dion3 f=.5": dict(rule="x_dion3", f=.5), "dion3 f=1": dict(rule="x_dion3", f=1)}
SWEEP = {f"c={c}": dict(rule="x_rand", shape=2, p3=round(1 - 2 * c, 2)) for c in (0, .1, .2, .3, .4, .5, .6, .7, .8)}  # c=.5 is Muon by SVD, the control
STEPS = {f"steps={n}": dict(rule="x_nsteps", n=n) for n in (1, 2, 3, 4, 5, 7, 10, 20)}  # steps=5 is Muon, the control
sig = lambda x: float(f"{x:.4g}")


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("base_file"); p.add_argument("--results", required=True); p.add_argument("--out", required=True); p.add_argument("--parts", default=""); p.add_argument("--only", default=""); p.add_argument("--sweep", action="store_true"); p.add_argument("--cs", default=""); p.add_argument("--steps", action="store_true"); p.add_argument("--par", type=int, default=9); p.add_argument("--hours", type=float, default=.1); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); os.makedirs(os.path.dirname(mp.PATH), exist_ok=True); L = [l for l in open(a.base_file) if '"base"' in l]; b = arms.parse(json.loads(([l for l in L if "gaps" in l] or L)[-1])["base"]); tune, held = [0, 1], [5, 6, 7, 8, 9]; R = mp.Runner(a.tier, a.par, a.hours); R.need([b], tune + held); B = np.array([R.val(b, w) for w in held]); print(f"{a.tier} base {B.mean():.4f}", flush=True); out = dict(tier=a.tier, base=arms.show(b), base_held=round(float(B.mean()), 4), rows={})
    for part, mods in PARTS.items():
        if a.parts and part not in a.parts.split(","): continue
        for name, k in (STEPS if a.steps else SWEEP if a.sweep else ARMS).items():
            if a.only and a.only not in name: continue
            if a.cs and a.sweep and name not in ["c=" + x for x in a.cs.split(",")]: continue
            make = lambda f, beta=.95: {m: dict(k, lr=sig(g["lr"] * f), beta=beta) if m in mods else g for m, g in b.items()}
            f = mp.walk(R, make, (.25, 1, 4), tune, f"{part} {name} lr x"); g2 = mp.walk(R, lambda x: make(f * x), (.63, 1, 1.6), tune, f"{part} {name} fine x"); f *= g2; opts = [make(f, x) for x in (.8, .9, .95)]; R.need(opts, tune); s = min(opts, key=lambda x: R.mean(x, tune)); R.need([s], held); g = B - np.array([R.val(s, w) for w in held])
            out["rows"][f"{part} {name}"] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / 5 ** .5), 4), wins=int((g > 0).sum()), f=sig(f), spec=arms.show(s)); print(f"{part} {name}: {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True); json.dump(out, open(a.out, "w"), indent=0)
    R.ex.shutdown()


if __name__ == "__main__":
    main()
