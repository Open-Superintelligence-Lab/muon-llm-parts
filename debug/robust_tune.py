"""Tunes an arm on the mean of 5 open worlds instead of one world, then scores it on 5 other open worlds that tuning never saw. One world's score moves about .003 from any small change, so a best setting picked there is partly luck. Only tune rows are read, and the sealed worlds are never touched.

Starts from the arm's joint row and changes each part's lr and momentum in turn. A stopped run continues by running the same command, because finished runs are skipped. Nothing here writes to choices.jsonl.
python3 debug/robust_tune.py muon-split muon --results path --seconds 500
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
import arms, env, runq, search
env.TIERS["long"] = dict(width=32, layers=2, heads=4, V=1000, batch=8, L=48, steps=1000, val_rows=3200, marks=5, bench=60)  # about 10 s, for slow momentum, which needs more steps than 5s has; workers import this file again, so they get it too
TUNE = (0, 1, 2, 3, 4); HELD = (5, 6, 7, 8, 9); BETAS = (.3, .5, .7, .8, .9, .95); PARTS = ("qk", "vo", "gu", "down", "emb", "head", "norm")


class Out(Exception): pass


def main():
    p = argparse.ArgumentParser(); p.add_argument("arms", nargs="+"); p.add_argument("--results", required=True); p.add_argument("--seconds", type=float, default=500); p.add_argument("--passes", type=int, default=2); p.add_argument("--tier", default="5s"); p.add_argument("--start", default=""); a = p.parse_args(); end = time.time() + a.seconds
    ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; job = lambda s, w: env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--world", str(w), "--evals", "tune", "--tag", "robust", "--results", a.results])

    def run(specs, worlds):
        left = runq.run([job(s, w) for s in specs for w in worlds], max(1, end - time.time()))
        if left: raise Out(left)
        rows = {}
        for l in open(a.results):
            r = json.loads(l); rows[(r["spec"], r["world"])] = r["scores"]["text.tune.loss"] if r["finite"] and not r["opt"]["unstable"] else 99.
        return [np.array([rows[(arms.show(s), w)] for w in worlds]) for s in specs]

    def mutate(s, m, **ch):
        t = {x: dict(g) for x, g in s.items()}; t[m].update(ch); return t

    def neighbours(s, m):
        g = s[m]; out = [mutate(s, m, lr=search.round_lr(g["lr"] * f)) for f in (.63, 1.6)]; k = search.knob_of(g["rule"])
        if k:
            b = g.get(k, .9 if g["rule"] == "adam" else .95); i = min(range(len(BETAS)), key=lambda j: abs(BETAS[j] - b))
            out += [mutate(s, m, **{k: BETAS[j]}) for j in (i - 1, i + 1) if 0 <= j < len(BETAS) and BETAS[j] != b]
        return out

    done = {}
    try:
        for arm in a.arms:
            start = arms.parse(a.start or [r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == arm and r["tier"] == "5s"][-1]["spec"]); s = start; best = run([s], TUNE)[0].mean(); n = 1; print(f"{arm}: start {best:.4f}", flush=True)
            for pas in range(a.passes):
                was = best
                for m in PARTS:
                    new = neighbours(s, m); v = [x.mean() for x in run(new, TUNE)]; n += len(new); i = int(np.argmin(v))
                    if v[i] < best - 1e-4: s, best = new[i], v[i]; print(f"  {m}: {best:.4f} {arms.show({m: s[m]})}", flush=True)
                if was - best < .0005: break
            done[arm] = (start, s); print(f"{arm}: tuned {best:.4f} after {n} settings\n  {arms.show(s)}", flush=True)
        names = list(done); h = {x: run(list(done[x]), HELD) for x in names}
        for x in names:
            g = h[x][0] - h[x][1]; print(f"{x} held-out worlds: one-world row {h[x][0].mean():.4f}, five-world row {h[x][1].mean():.4f}, gain {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
        for x in names[1:]:
            g = h[x][1] - h[names[0]][1]; print(f"{names[0]} over {x}, five-world rows, held-out worlds: {np.round(g, 4).tolist()} mean {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
    except Out as e: print(f"{e} runs left: rerun the same command", flush=True)
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
