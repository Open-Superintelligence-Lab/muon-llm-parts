"""Is query and key's indifference to Muon caused by QK-norm? Repeats the query-key and value-output tests on the same model with QK-norm removed. The baseline's Muon lrs are walked again without QK-norm, first together and then on q,k alone. Settings are chosen on tune worlds and scored on other worlds, on tune rows. Positive gap beats the baseline without QK-norm.

python3 debug/noqk_parts.py 5s results/add/w48_out.jsonl --results results/noqk/w48.jsonl [--par 4] [--hours H] [--out out.jsonl]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

QK, VO = ("q", "k"), ("v", "o"); plain_job = mp.job
ADAM = (.004, .01, .025)


def job(s, tier, w):
    a = plain_job(s, tier, w); a.qknorm = False; return a


mp.job = job


def adam(b, parts, lr): return {m: dict(rule="adam", lr=lr) if m in parts else g for m, g in b.items()}
def spec(b, parts, knobs, f): return {m: dict(rule="x_muon_spec", lr=round(g["lr"] * f, 5), beta=g["beta"], **knobs) if m in parts else g for m, g in b.items()}
def shape(b, parts, knobs, f): return {m: dict(rule="x_muon_shape", lr=round(g["lr"] * f, 5), beta=g["beta"], h=1, **knobs) if m in parts else g for m, g in b.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("base_file"); p.add_argument("--results", required=True); p.add_argument("--par", type=int, default=4); p.add_argument("--hours", type=float, default=11); p.add_argument("--out", default=""); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); os.makedirs(os.path.dirname(mp.PATH), exist_ok=True); b0 = arms.parse([json.loads(l) for l in open(a.base_file) if "gaps" in l][-1]["base"]); tune, held = [0, 1], [5, 6, 7, 8, 9]; R = mp.Runner(a.tier, a.par, a.hours)
    f = mp.walk(R, lambda f: mp.scaled(b0, mp.MATS, f), (.63, 1, 1.6), tune, "all Muon lrs x"); b1 = mp.scaled(b0, mp.MATS, f); f = mp.walk(R, lambda f: mp.scaled(b1, QK, f), (.63, 1, 1.6), tune, "q,k Muon lrs x"); b = mp.scaled(b1, QK, f)
    R.need([b], tune + held); B = np.array([R.val(b, w) for w in held]); print(f"{a.tier} no QK-norm base held-out {B.mean():.4f}", flush=True); out = dict(tier=a.tier, base=arms.show(b), base_held=round(float(B.mean()), 4), rows={})
    tests = dict(adam_qk=(lambda f: adam(b, QK, f), ADAM), adam_vo=(lambda f: adam(b, VO, f), ADAM), grad_qk=(lambda f: spec(b, QK, dict(p=1), f), (.63, 1, 1.6)), grad_vo=(lambda f: spec(b, VO, dict(p=1), f), (.63, 1, 1.6)), inv5_qk=(lambda f: spec(b, QK, dict(p=-.5), f), (.63, 1, 1.6)), weak_qk=(lambda f: shape(b, QK, dict(top=1, weak=0), f), (.63, 1, 1.6)), weak_vo=(lambda f: shape(b, VO, dict(top=1, weak=0), f), (.63, 1, 1.6)))
    for name, (make, grid) in tests.items():
        f = mp.walk(R, make, grid, tune, f"{name} setting"); s = make(f); R.need([s], held); g = B - np.array([R.val(s, w) for w in held])
        out["rows"][name] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(g) ** .5), 4), wins=int((g > 0).sum()), setting=f, spec=arms.show(s)); print(f"{name}: {g.mean():+.4f} error {g.std(ddof=1) / len(g) ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
    if a.out: open(a.out, "a").write(json.dumps(out) + "\n")
    R.ex.shutdown()


if __name__ == "__main__":
    main()
