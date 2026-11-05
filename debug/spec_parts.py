"""Which step shapes does each group accept? Keeps Muon's singular directions and changes only their sizes in one group, from a tuned whole-matrix Muon baseline. Settings are chosen on tune worlds and scored on other worlds, on tune rows. Positive gap beats the baseline.

python3 debug/spec_parts.py 5s results/add/w48_out.jsonl --results results/spec/w48.jsonl [--seed results/add/w48.jsonl] [--par 4]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

GROUPS = dict(vo=("v", "o"), qk=("q", "k"), ffn=("gate", "up", "down"), all=mp.MATS); KIND = dict(flat=dict(p=0), half=dict(p=.5), grad=dict(p=1), rev=dict(p=1, rev=1), rand=dict(rand=1), inv5=dict(p=-.5), inv=dict(p=-1)); ONLY_ALL = ("flat", "half")


def changed(b, parts, knobs, f): return {m: dict(rule="x_muon_spec", lr=round(g["lr"] * f, 5), beta=g["beta"], **knobs) if m in parts else g for m, g in b.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("base_file"); p.add_argument("--results", required=True); p.add_argument("--seed", default=""); p.add_argument("--par", type=int, default=4); p.add_argument("--hours", type=float, default=11); p.add_argument("--out", default=""); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); os.makedirs(os.path.dirname(mp.PATH), exist_ok=True); b = arms.parse([json.loads(l) for l in open(a.base_file) if "gaps" in l][-1]["base"]); tune, held = [0, 1], [5, 6, 7, 8, 9]
    if a.seed and not os.path.exists(mp.PATH):
        want = {arms.show(b)}
        with open(mp.PATH, "w") as f: f.writelines(l for l in open(a.seed) if json.loads(l)["spec"] in want)
    R = mp.Runner(a.tier, a.par, a.hours); R.need([b], tune + held); B = np.array([R.val(b, w) for w in held]); print(f"{a.tier} base held-out {B.mean():.4f}", flush=True); out = dict(tier=a.tier, base=arms.show(b), base_held=round(float(B.mean()), 4), rows={})
    for kind, knobs in KIND.items():
        for name, parts in GROUPS.items():
            if kind in ONLY_ALL and name != "all": continue
            f = mp.walk(R, lambda f: changed(b, parts, knobs, f), (.63, 1, 1.6), tune, f"{kind} {name} lr x"); s = changed(b, parts, knobs, f); R.need([s], held); g = B - np.array([R.val(s, w) for w in held])
            out["rows"][f"{kind}_{name}"] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(g) ** .5), 4), wins=int((g > 0).sum()), lr_factor=f, spec=arms.show(s)); print(f"{kind} {name}: {g.mean():+.4f} error {g.std(ddof=1) / len(g) ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
    if a.out: open(a.out, "a").write(json.dumps(out) + "\n")
    R.ex.shutdown()


if __name__ == "__main__":
    main()
