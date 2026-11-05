"""Candidate optimizer: each group gets its own step shape. For each group it picks the power p and lr factor on tune worlds, starting from a tuned whole-matrix Muon baseline, then scores each group alone and all three together on other worlds, on tune rows. Positive gap beats the baseline.

python3 debug/pick_shape.py 5s results/add/w48_out.jsonl --results results/spec/pick_w48.jsonl [--seed results/add/w48.jsonl] [--par 4] [--hours H] [--out out.jsonl]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

GROUPS = dict(qk=(("q", "k"), (.1, .25, .5, 1)), vo=(("v", "o"), (-.25, -.1, .1)), ffn=(("gate", "up", "down"), (-.25, -.1, .1)))


def shaped(b, parts, p, f): return {m: dict(rule="x_muon_spec", lr=round(g["lr"] * f, 5), beta=g["beta"], p=p) if m in parts else g for m, g in b.items()}


def main():
    a0 = argparse.ArgumentParser(); a0.add_argument("tier"); a0.add_argument("base_file"); a0.add_argument("--results", required=True); a0.add_argument("--seed", default=""); a0.add_argument("--par", type=int, default=4); a0.add_argument("--hours", type=float, default=11); a0.add_argument("--out", default=""); a = a0.parse_args()
    mp.PATH = os.path.abspath(a.results); os.makedirs(os.path.dirname(mp.PATH), exist_ok=True); b = arms.parse([json.loads(l) for l in open(a.base_file) if "gaps" in l][-1]["base"]); tune, held = [0, 1], [5, 6, 7, 8, 9]
    if a.seed and not os.path.exists(mp.PATH):
        want = {arms.show(b)}
        with open(mp.PATH, "w") as f: f.writelines(l for l in open(a.seed) if json.loads(l)["spec"] in want)
    R = mp.Runner(a.tier, a.par, a.hours); R.need([b], tune + held); B = np.array([R.val(b, w) for w in held]); print(f"{a.tier} base tune {R.mean(b, tune):.4f} held-out {B.mean():.4f}", flush=True); out = dict(tier=a.tier, base=arms.show(b), base_held=round(float(B.mean()), 4), rows={}); joint = b; picks = {}
    for name, (parts, powers) in GROUPS.items():
        best = None
        for p in powers:
            f = mp.walk(R, lambda f: shaped(b, parts, p, f), (.63, 1, 1.6), tune, f"{name} p={p} lr x"); v = R.mean(shaped(b, parts, p, f), tune)
            if best is None or v < best[0]: best = (v, p, f)
        keep = best[0] < R.mean(b, tune); picks[name] = dict(p=best[1], lr_factor=best[2], tune=round(best[0], 4), kept=bool(keep)); s = shaped(b, parts, best[1], best[2]); R.need([s], held); g = B - np.array([R.val(s, w) for w in held])
        out["rows"][name] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(g) ** .5), 4), wins=int((g > 0).sum()), **picks[name]); print(f"{name} p={best[1]} lr x{best[2]:g} kept {keep}: {g.mean():+.4f} error {g.std(ddof=1) / len(g) ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
        if keep: joint = shaped(joint, parts, best[1], best[2])
    R.need([joint], held); g = B - np.array([R.val(joint, w) for w in held]); out["rows"]["joint"] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(g) ** .5), 4), wins=int((g > 0).sum()), spec=arms.show(joint)); print(f"joint: {g.mean():+.4f} error {g.std(ddof=1) / len(g) ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
    if a.out: open(a.out, "a").write(json.dumps(out) + "\n")
    R.ex.shutdown()


if __name__ == "__main__":
    main()
