"""Compares chosen settings with tuned Muon on the 5 open worlds, on their tune rows. It never touches the sealed worlds or report rows. A setting is name:part:lr-multiplier.

python3 debug/open_worlds.py 5s qk x_head_trust:1 --base muon-split --results path
python3 debug/open_worlds.py 5s - --base muon --arms muon-split --results path    whole tuned arms
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
import arms, env, runq, search
WORLDS = (1, 2, 3, 4, 8)


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("part"); p.add_argument("picks", nargs="*"); p.add_argument("--results", required=True); p.add_argument("--seconds", type=float, default=230); p.add_argument("--base", default="muon"); p.add_argument("--arms", default=""); a = p.parse_args()
    ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; tuned = lambda n: arms.parse([r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == n and r["tier"] == "5s"][-1]["spec"]); base = tuned(a.base); specs = {a.base: base}
    for n in filter(None, a.arms.split(",")): specs[n] = tuned(n)
    for x in a.picks if a.part != "-" else []:
        n, f = x.split(":"); s = {k: dict(g) for k, g in base.items()}
        for m in a.part.split(","): s[m] = dict(rule=n, lr=search.round_lr(base[m]["lr"] * float(f)))
        specs[x] = s
    job = lambda s, w: env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--world", str(w), "--evals", "tune", "--tag", "open", "--results", a.results])
    left = runq.run([job(s, w) for s in specs.values() for w in WORLDS], a.seconds)
    if runq.EX: runq.EX.shutdown(); runq.EX = None
    if left: print("left", left)
    rows = {}
    for l in open(a.results):
        r = json.loads(l); rows[(r["spec"], r["world"])] = r["scores"]["text.tune.loss"] if r["finite"] and not r["opt"]["unstable"] else float("inf")
    for x, s in specs.items():
        if x == a.base: continue
        g = np.array([rows[(arms.show(base), w)] - rows[(arms.show(s), w)] for w in WORLDS]); print(f"{x}: gaps to {a.base} {np.round(g, 4).tolist()} mean {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)


if __name__ == "__main__":
    main()
