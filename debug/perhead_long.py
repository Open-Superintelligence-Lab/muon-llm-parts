"""Per-head Muon on query and key (muonsplit, h heads, lr x f of whole-matrix Muon's) against the tuned base, on many worlds at any tier.

python3 debug/perhead_long.py --tier 5sx16 --base results/add/w48x4000_out.jsonl --results results/papers/w48x4000_band.jsonl --out results/papers/perhead_4000.json [--fs 2,4] [--worlds 5,...,14]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms, env


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", required=True); p.add_argument("--base", required=True); p.add_argument("--results", required=True); p.add_argument("--out", required=True); p.add_argument("--fs", default="2,4"); p.add_argument("--worlds", default="5,6,7,8,9,10,11,12,13,14"); p.add_argument("--par", type=int, default=9); p.add_argument("--hours", type=float, default=.15); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); W = [int(x) for x in a.worlds.split(",")]; heads = env.TIERS[a.tier]["heads"]
    b = arms.parse((lambda L: ([x for x in L if "gaps" in x] or L)[-1])([json.loads(l) for l in open(a.base) if '"base"' in l])["base"])
    R = mp.Runner(a.tier, a.par, a.hours); R.need([b], W); B = np.array([R.val(b, w) for w in W])
    specs = {float(f): mp.split(b, heads, float(f)) for f in a.fs.split(",")}; R.need(list(specs.values()), W)
    out = dict(tier=a.tier, worlds=W, base=round(float(B.mean()), 4), rows=[])
    for f, s in specs.items():
        g = B - np.array([R.val(s, w) for w in W]); out["rows"].append(dict(f=f, gain=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(W) ** .5), 4), wins=int((g > 0).sum())))
        print(f"per-head q,k lr x{f}: {g.mean():+.4f} error {g.std(ddof=1) / len(W) ** .5:.4f} wins {(g > 0).sum()}/{len(W)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=0); R.ex.shutdown()


if __name__ == "__main__":
    main()
