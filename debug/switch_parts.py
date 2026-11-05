"""When does a group need Muon? Changes step shape once during training, on one group, each shape at its own tuned lr factor from spec_parts. Scored on worlds 5-9, tune rows.

python3 debug/switch_parts.py 5sx4 1000 results/add/w48x1000_out.jsonl results/spec/w48x1000_out.jsonl --results results/spec/switch_w48x1000.jsonl --seed results/spec/w48x1000.jsonl
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

GROUPS = dict(vo=("v", "o"))


def switched(b, parts, f, at, first):
    p1, p2, f1, f2 = (1, 0, f, 1) if first == "grad" else (0, 1, 1, f)
    return {m: dict(rule="x_muon_switch", lr=g["lr"], beta=g["beta"], at=at, p1=p1, p2=p2, f1=f1, f2=f2) if m in parts else g for m, g in b.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("steps", type=int); p.add_argument("base_file"); p.add_argument("spec_file"); p.add_argument("--results", required=True); p.add_argument("--seed", default=""); p.add_argument("--par", type=int, default=4); p.add_argument("--hours", type=float, default=1); p.add_argument("--out", default=""); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); os.makedirs(os.path.dirname(mp.PATH), exist_ok=True); b = arms.parse([json.loads(l) for l in open(a.base_file) if "gaps" in l][-1]["base"]); rows = [json.loads(l) for l in open(a.spec_file)][-1]["rows"]; held = [5, 6, 7, 8, 9]; out = dict(tier=a.tier, rows={})
    for name, parts in GROUPS.items():
        whole = arms.parse(rows[f"grad_{name}"]["spec"]); f = rows[f"grad_{name}"]["lr_factor"]
        if a.seed and not os.path.exists(mp.PATH):
            want = {arms.show(b), arms.show(whole)}
            with open(mp.PATH, "w") as o: o.writelines(l for l in open(a.seed) if json.loads(l)["spec"] in want)
        R = mp.Runner(a.tier, a.par, a.hours); R.need([b, whole], held); B = np.array([R.val(b, w) for w in held]); g = B - np.array([R.val(whole, w) for w in held]); print(f"{a.tier} base {B.mean():.4f}; gradient's shape throughout on {name}: {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f}", flush=True)
        for first in ("grad", "muon"):
            for frac in (.25, .5, .75):
                s = switched(b, parts, f, int(a.steps * frac), first); R.need([s], held); g = B - np.array([R.val(s, w) for w in held]); out["rows"][f"{name}_{first}_first_{frac}"] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / 5 ** .5), 4), wins=int((g > 0).sum()), spec=arms.show(s)); print(f"{name} {first} first, switch at {frac}: {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
    if a.out: open(a.out, "a").write(json.dumps(out) + "\n")
    R.ex.shutdown()


if __name__ == "__main__":
    main()
