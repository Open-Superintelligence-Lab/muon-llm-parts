"""Compares chosen settings with the tuned --base arm on the 5 open worlds, on tune rows. The base was picked on world 0 from hundreds of settings, so its world-0 score flatters it; these worlds do not. Each setting is rule:lr-multiplier[:knob=value,...] on the listed parts, with parts at the base's momentum unless a knob says otherwise.

python3 debug/open_pick.py qk,vo,gu,down x_franken:1:warm=25 x_weak_memory:1.6:ns=1,fast=.7 --results path
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
import arms, env, runq, sandbox, search
WORLDS = (1, 2, 3, 4, 8)


def main():
    p = argparse.ArgumentParser(); p.add_argument("parts"); p.add_argument("picks", nargs="+"); p.add_argument("--base", default="muon-split"); p.add_argument("--results", required=True); p.add_argument("--seconds", type=float, default=200); p.add_argument("--tier", default="5s"); a = p.parse_args()
    ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; base = arms.parse([r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == a.base and r["tier"] == "5s"][-1]["spec"]); specs = {}
    for x in a.picks:
        bits = x.split(":"); s = {m: dict(g) for m, g in base.items()}; kn = {t.split("=")[0]: float(t.split("=")[1]) for t in bits[2].split(",")} if len(bits) > 2 else {}; sandbox.load(bits[0])
        for m in a.parts.split(","): s[m] = dict(dict(rule=bits[0], lr=search.round_lr(base[m]["lr"] * float(bits[1])), beta=base[m].get("beta", .95), h=4 if base[m]["rule"] == "muonsplit" else 1), **kn)
        specs[x] = s
    job = lambda s, w: env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--world", str(w), "--evals", "tune", "--tag", "open", "--results", a.results])
    left = runq.run([job(s, w) for s in [base] + list(specs.values()) for w in WORLDS], a.seconds)
    if runq.EX: runq.EX.shutdown()
    if left: print("left", left, "rerun the same command")
    rows = {}
    for l in open(a.results):
        r = json.loads(l); rows[(r["spec"], r["world"])] = r
    val = lambda s, w: (lambda r: r["scores"]["text.tune.loss"] if r["finite"] and not r["opt"]["unstable"] else float("inf"))(rows[(arms.show(s), w)])
    for x, s in specs.items():
        g = np.array([val(base, w) - val(s, w) for w in WORLDS]); t = np.mean([rows[(arms.show(s), w)]["s_per_step"] / rows[(arms.show(base), w)]["s_per_step"] for w in WORLDS]); b = rows[(arms.show(s), 1)]["state_bytes"] / rows[(arms.show(base), 1)]["state_bytes"]
        print(f"{x}: {np.round(g, 4).tolist()} mean {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5, time {t:.2f}x, state {b:.2f}x", flush=True)


if __name__ == "__main__":
    main()
