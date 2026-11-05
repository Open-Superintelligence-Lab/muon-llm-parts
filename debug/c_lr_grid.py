"""Is the exponent's gain the spectrum shape, or a different best step size? Loss on worlds 5-9 for exact (GG^T)^-c G on one part (x_rand shape 2, resized to Muon's Frobenius size, so the same lr gives the same update size for every c) over a grid of c and lr multipliers of the tuned Muon lr. Positive gain beats tuned Muon.

python3 debug/c_lr_grid.py --part vo --results results/papers/w48x1000.jsonl --out results/papers/c_lr_vo.json [--cs .3,.5,.7] [--fs .25,.5,1] [--par 9] [--hours .1] [--worlds 5,6,7,8,9]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

PARTS = dict(vo=("v", "o"), qk=("q", "k"), ffn=("gate", "up", "down")); sig = lambda x: float(f"{x:.4g}")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--part", required=True); p.add_argument("--results", required=True); p.add_argument("--out", required=True); p.add_argument("--cs", default=".3,.4,.5,.6,.7,.8"); p.add_argument("--fs", default=".25,.35,.5,.7,1"); p.add_argument("--beta", type=float, default=.9); p.add_argument("--par", type=int, default=9); p.add_argument("--hours", type=float, default=.1); p.add_argument("--worlds", default="5,6,7,8,9"); p.add_argument("--muon", action="store_true", help="control: Muon itself (Newton-Schulz) at each lr and --beta; c is ignored"); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); b = arms.parse([json.loads(l) for l in open("results/add/w48x1000_out.jsonl") if "gaps" in l][-1]["base"]); held = [int(x) for x in a.worlds.split(",")]; mods = PARTS[a.part]
    R = mp.Runner("5sx4", a.par, a.hours); R.need([b], held); B = np.array([R.val(b, w) for w in held])
    grid = {(c, f): {m: (dict(g, lr=sig(g["lr"] * f), beta=a.beta) if a.muon else dict(rule="x_rand", shape=2, p3=round(1 - 2 * c, 2), beta=a.beta, lr=sig(g["lr"] * f), h=4)) if m in mods else g for m, g in b.items()} for c in map(float, a.cs.split(",")) for f in map(float, a.fs.split(","))}
    specs = list(grid.values())
    for i in range(0, len(specs), 9): R.need(specs[i:i + 9], held)
    out = dict(part=a.part, worlds=held, base_held=round(float(B.mean()), 4), rows=[])
    for (c, f), s in grid.items():
        g = B - np.array([R.val(s, w) for w in held]); out["rows"].append(dict(c=c, f=f, gain=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(held) ** .5), 4), wins=int((g > 0).sum())))
    cs = sorted({r["c"] for r in out["rows"]}); fs = sorted({r["f"] for r in out["rows"]}); G = {(r["c"], r["f"]): r["gain"] for r in out["rows"]}
    print(f"{a.part}: gain over tuned Muon, rows c, columns lr x " + " ".join(f"{f:>7}" for f in fs))
    for c in cs: print(f"c={c:<4}" + " ".join(f"{G[(c, f)]:+.4f}" for f in fs))
    json.dump(out, open(a.out, "w"), indent=0); R.ex.shutdown()


if __name__ == "__main__":
    main()
