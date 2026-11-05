"""For each part, does the optimizer matter, and is a better one known? Moves one part from the tuned baseline to one other published optimizer, tunes lr then momentum on worlds 0,1, scores on worlds 5-9, tune rows. Frozen trains the part not at all.

python3 debug/panel_parts.py 5s results/add/w48_out.jsonl --results results/panel/w48.jsonl --seed results/add/w48.jsonl --out results/panel/w48_out.json [--parts qk,vo] [--rules adam,lion] [--par 6] [--hours H]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

MAT = ("adam", "sgdm", "lion", "rownorm", "normsgd"); PARTS = dict(vo=(("v", "o"), MAT + ("muon", "muonsplit")), qk=(("q", "k"), MAT + ("muon", "muonsplit")), ffn=(("gate", "up", "down"), MAT + ("muon",)), emb=(("emb",), ("adam", "muon", "sgdm", "lion", "rownorm")), head=(("head",), ("adam", "muon", "sgdm", "lion", "rownorm")), norm=(("norm",), ("adam", "sgdm", "lion", "norm1d")))
sig = lambda x: float(f"{x:.4g}")


def moved(b, parts, rule, lr, **k): return {m: dict(rule=rule, lr=sig(lr), **k) if m in parts else g for m, g in b.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("base_file"); p.add_argument("--results", required=True); p.add_argument("--seed", default=""); p.add_argument("--out", default=""); p.add_argument("--parts", default=""); p.add_argument("--rules", default=""); p.add_argument("--par", type=int, default=6); p.add_argument("--hours", type=float, default=1); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); os.makedirs(os.path.dirname(mp.PATH), exist_ok=True); b = arms.parse([json.loads(l) for l in open(a.base_file) if "gaps" in l][-1]["base"]); tune, held = [0, 1], [5, 6, 7, 8, 9]
    if a.seed and not os.path.exists(mp.PATH):
        with open(mp.PATH, "w") as f: f.writelines(l for l in open(a.seed) if json.loads(l)["spec"] == arms.show(b))
    R = mp.Runner(a.tier, a.par, a.hours); R.need([b], tune + held); B = np.array([R.val(b, w) for w in held]); print(f"{a.tier} base {B.mean():.4f}", flush=True); out = dict(tier=a.tier, base=arms.show(b), base_held=round(float(B.mean()), 4), rows={})

    def score(name, s, **extra):
        R.need([s], held); g = B - np.array([R.val(s, w) for w in held]); out["rows"][name] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / 5 ** .5), 4), wins=int((g > 0).sum()), spec=arms.show(s), **extra); print(f"{name}: {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
        if a.out: json.dump(out, open(a.out, "w"), indent=0)

    for name, (parts, rules) in PARTS.items():
        if a.parts and name not in a.parts.split(","): continue
        if not a.rules: score(f"{name} frozen", moved(b, parts, "sgdm", 0))
        for rule in rules:
            if a.rules and rule not in a.rules.split(","): continue
            if b[parts[0]]["rule"] == rule: continue
            lr = arms.LRS[rule]; f = mp.walk(R, lambda f: moved(b, parts, rule, lr * f), (.25, 1, 4), tune, f"{name} {rule} lr x"); lr *= f; f = mp.walk(R, lambda f: moved(b, parts, rule, lr * f), (.63, 1, 1.6), tune, f"{name} {rule} fine x"); lr *= f
            knob, grid = ("b1", (.7, .9)) if rule == "adam" else ("beta", (.8, .9, .95)); opts = [moved(b, parts, rule, lr, **{knob: x}) for x in grid]; R.need(opts, tune); best = min(opts, key=lambda s: R.mean(s, tune))
            score(f"{name} {rule}", best, lr=sig(lr), tune_edge=bool(R.mean(best, tune) >= 99))
    R.ex.shutdown()


if __name__ == "__main__":
    main()
