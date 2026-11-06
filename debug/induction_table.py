"""Per arm and world: final loss, copy score (ICL, mean of the last 5 probes), best-head induction attention at the end, and the first step it reaches 0.05.
python3 debug/induction_table.py results/induction/w64rep25x4000_w5.json results/induction/w64rep25x4000_w67.json ..."""
import json, sys, numpy as np
R = [r for f in sys.argv[1:] for r in json.load(open(f))["runs"]]
att = lambda x: max(max(h) for h in x["att"])
print(f"{'arm':16s} " + "  ".join(f"{'world ' + str(w):>27s}" for w in sorted({r['world'] for r in R})) + "   mean loss  mean icl")
for n in dict.fromkeys(r["name"] for r in R):
    rs = sorted([r for r in R if r["name"] == n], key=lambda r: r["world"]); cells = []
    for r in rs:
        c = r["curve"]; f = next((x["step"] for x in c if att(x) >= 0.05), None)
        cells.append(f"{c[-1]['loss']:.4f} {np.mean([x['icl'] for x in c[-5:]]):.3f} {att(c[-1]):.3f} {f if f is not None else '-':>5}" + ("" if r["step"] == c[-1]["step"] and not r["stopped_early"] else "*"))
    print(f"{n:16s} " + "  ".join(f"{s:>27s}" for s in cells) + f"   {np.mean([r['curve'][-1]['loss'] for r in rs]):.4f}    {np.mean([np.mean([x['icl'] for x in r['curve'][-5:]]) for r in rs]):.3f}")
print("cell: final loss, ICL (last 5 probes), best-head attention at end, first step attention >= 0.05; * = unfinished")
