"""Which quarter of the spectrum carries the exponent's gain on one part? Runs optimizers/x_band.py (exact polar with a set step per quarter of singular directions, t1 top) at Muon's tuned lr for that part, for each set of quarter steps, and prints gain over tuned Muon on the given worlds.

python3 debug/band_grid.py --part vo --bands ".32,.48,.70,1.77 .5,.5,.5,.5" --results results/papers/w48x1000.jsonl --out results/papers/band_vo.json [--beta .95] [--worlds 5,...,14] [--tier 5sx4]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

PARTS = dict(vo=("v", "o"), qk=("q", "k"), ffn=("gate", "up", "down"))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--part", required=True); p.add_argument("--bands", required=True); p.add_argument("--results", required=True); p.add_argument("--out", required=True); p.add_argument("--beta", type=float, default=.95); p.add_argument("--worlds", default="5,6,7,8,9,10,11,12,13,14"); p.add_argument("--tier", default="5sx4"); p.add_argument("--par", type=int, default=9); p.add_argument("--hours", type=float, default=.075); p.add_argument("--flip4", type=int, default=0); p.add_argument("--base", default="results/add/w48x1000_out.jsonl"); p.add_argument("--switch", default="", help="at,u1,u2,u3,u4: after step at, use quarter steps u in place of the bands"); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); W = [int(x) for x in a.worlds.split(",")]; mods = PARTS[a.part]
    b = arms.parse((lambda L: ([x for x in L if "gaps" in x] or L)[-1])([json.loads(l) for l in open(a.base) if '"base"' in l])["base"])
    R = mp.Runner(a.tier, a.par, a.hours); R.need([b], W); B = np.array([R.val(b, w) for w in W])
    bands = [tuple(float(x) for x in s.split(",")) for s in a.bands.split()]
    grid = {t: {m: dict(rule="x_band", lr=g["lr"], beta=a.beta, t1=t[0], t2=t[1], t3=t[2], t4=t[3], **({"flip4": 1} if a.flip4 else {}), **(dict(zip(("at", "u1", "u2", "u3", "u4"), [int(float(x)) if i == 0 else float(x) for i, x in enumerate(a.switch.split(","))])) if a.switch else {})) if m in mods else g for m, g in b.items()} for t in bands}
    specs = list(grid.values())
    for i in range(0, len(specs), 9): R.need(specs[i:i + 9], W)
    old = json.load(open(a.out)) if os.path.exists(a.out) else dict(part=a.part, worlds=W, base=round(float(B.mean()), 4), rows=[])
    for t, s in grid.items():
        g = B - np.array([R.val(s, w) for w in W]); row = dict(tier=a.tier, t=list(t), beta=a.beta, flip4=a.flip4, switch=a.switch, gain=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(W) ** .5), 4), wins=int((g > 0).sum()))
        old["rows"] = [r for r in old["rows"] if (r.get("tier", "5sx4"), r["t"], r["beta"], r.get("flip4", 0), r.get("switch", "")) != (a.tier, row["t"], row["beta"], a.flip4, a.switch)] + [row]
        print(f"{a.part} quarters {t} momentum {a.beta}{' random signs in quarter 4' if a.flip4 else ''}{' then ' + a.switch if a.switch else ''}: {row['gain']:+.4f} error {row['error']:.4f} wins {row['wins']}/{len(W)}", flush=True)
    json.dump(old, open(a.out, "w"), indent=0); R.ex.shutdown()


if __name__ == "__main__":
    main()
