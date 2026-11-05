"""Do gains picked on worlds 5-9 hold on other open worlds? Reads the chosen specs stored in papers_parts json files (rows with gain above --min) and scores them against tuned Muon on worlds 10-14, which no choice in this repo looked at. Sealed worlds stay unread.

python3 debug/recheck_worlds.py results/papers/dion3.json results/papers/hyc.json --results results/papers/w48x1000.jsonl --out results/papers/recheck.json [--min 0] [--worlds 10,11,12,13,14]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms


def main():
    p = argparse.ArgumentParser(); p.add_argument("files", nargs="+"); p.add_argument("--results", required=True); p.add_argument("--out", required=True); p.add_argument("--min", type=float, default=0); p.add_argument("--worlds", default="10,11,12,13,14"); p.add_argument("--par", type=int, default=9); p.add_argument("--hours", type=float, default=.1); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); W = [int(x) for x in a.worlds.split(",")]
    b = arms.parse([json.loads(l) for l in open("results/add/w48x1000_out.jsonl") if "gaps" in l][-1]["base"])
    picks = {}
    for f in a.files:
        d = json.load(open(f))
        for name, r in d["rows"].items():
            if r["mean"] > a.min and d.get("tier", "5sx4") == "5sx4": picks[f"{os.path.basename(f)[:-5]} {name}"] = (r, arms.parse(r["spec"]))
    R = mp.Runner("5sx4", a.par, a.hours); R.need([b], W); B = np.array([R.val(b, w) for w in W])
    specs = [s for _, s in picks.values()]
    for i in range(0, len(specs), 9): R.need(specs[i:i + 9], W)
    out = dict(worlds=W, base=round(float(B.mean()), 4), rows={})
    for name, (r, s) in picks.items():
        g = B - np.array([R.val(s, w) for w in W]); out["rows"][name] = dict(old=r["mean"], new=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(W) ** .5), 4), wins=int((g > 0).sum()))
        print(f"{name}: worlds 5-9 {r['mean']:+.4f}, worlds {a.worlds} {g.mean():+.4f} error {g.std(ddof=1) / len(W) ** .5:.4f} wins {(g > 0).sum()}/{len(W)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=0); R.ex.shutdown()


if __name__ == "__main__":
    main()
