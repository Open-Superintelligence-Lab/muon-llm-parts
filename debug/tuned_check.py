"""Is each tuned baseline really tuned? From its chosen setting, moves one thing at a time on tune rows of world 0: each module's lr by 1.5x either way, then momentum for the whole arm. Prints what any move gains.

python3 debug/tuned_check.py 5s [--results tune_results.jsonl] [--seconds 240]
"""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import arms, env, runq, search

MOMENTUM = dict(beta=(.8, .9, .98), b1=(.8, .95), b2=(.9, .99, .999))


def lr_moves(spec):
    return [(f"{m} lr x{f:.2g}", search.with_lr(spec, m, search.round_lr(spec[m]["lr"] * f))) for m in arms.MODULES for f in (1.5, 1 / 1.5)]


def momentum_moves(spec):
    """One value for the whole arm, on the modules whose rule reads that knob. Adam reads b1 and b2, every other rule reads beta."""
    out = []
    for knob, vals in MOMENTUM.items():
        for v in vals:
            s = {k: dict(g) for k, g in spec.items()}; hit = False
            for m in arms.MODULES:
                if (s[m]["rule"] == "adam") == (knob != "beta"): s[m][knob] = v; hit = True
            if hit: out.append((f"{knob}={v:g}", s))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("--results", default=search.RESULTS); p.add_argument("--seconds", type=float, default=240); a = p.parse_args()
    x = env.parser().parse_args([]); x.__dict__.update(tier=a.tier, results=a.results, tag="search", moe=0, tied=False, qknorm=True); book = search.Book(os.path.join(HERE, a.results)); budget = time.time() + a.seconds
    picks = [r for r in search.choices(argparse.Namespace(choices="choices.jsonl")) if r["kind"] == "tune" and r.get("joint") and r["tier"] == ("5s" if a.tier == "tiny" else a.tier) and r["moe"] == 0]
    out = []
    for r in sorted(picks, key=lambda r: r["tune"]):
        spec = {k: g for k, g in arms.parse(r["spec"]).items() if k in arms.MODULES}; spec = dict(arms.parse(r["spec"])); t0 = time.time()
        res = search.try_all([spec], x, book, budget); base = res[0][0]; moves = lr_moves(spec) + momentum_moves(spec)
        vals = search.try_all([s for _, s in moves], x, book, budget); got = sorted(((v, n) for (v, _), (n, _) in zip(vals, moves)), key=lambda p: p[0])
        lrs = [(v, n) for v, n in got if " lr " in n]; mom = [(v, n) for v, n in got if " lr " not in n]; row = dict(arm=r["arm"], tune=round(base, 4), lr_gain=round(base - lrs[0][0], 4), lr_move=lrs[0][1], mom_gain=round(base - mom[0][0], 4), mom_move=mom[0][1], s=round(time.time() - t0)); out.append(row); print(json.dumps(row), flush=True)
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
