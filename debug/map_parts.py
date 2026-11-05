"""Maps where Muon matters: starts from a tuned Muon baseline and gives one matrix at a time to Adam. Every setting is chosen on tune worlds and scored on other worlds, on tune rows; sealed worlds are never read. Positive gap means the change beats Muon.

python3 debug/map_parts.py w128 --results map.jsonl [--tune 0,1] [--held 5,6,7,8,9] [--par 4] [--hours 11]
Finished runs are skipped, so the same command continues a stopped map. --time-only prints seconds per run and stops.
"""
import argparse, concurrent.futures as cf, json, multiprocessing as mp, os, re, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np, torch
import arms, env, runq

SHAPE = dict(V=4000, batch=64, L=128, val_rows=3200, marks=8, bench=60, own_shard=True)  # only width, depth and steps change along the map
for name, width, layers, heads, steps in (("w64", 64, 4, 4, 2000), ("w128", 128, 6, 8, 2000), ("w256", 256, 6, 8, 2000), ("w512", 512, 8, 8, 2000), ("w64x500", 64, 4, 4, 500), ("w64x8k", 64, 4, 4, 8000)):
    env.TIERS[name] = dict(SHAPE, width=width, layers=layers, heads=heads, steps=steps)
env.TIERS["dbg"] = dict(width=16, layers=1, heads=2, V=2000, batch=4, L=24, steps=20, val_rows=12, marks=2, bench=4, min_facts=4)  # checks code only

old_module_of = arms.module_of
arms.module_of = lambda n, w: (lambda m: m.group(1) if m else old_module_of(n, w))(re.search(r"\.(q|k|v|o|gate|up)(\.\d+)?\.weight$", n))  # q, k, v, o, gate and up get their own settings; arms.py stays as it is and workers inherit this

MUON = "down=muon:0.051:beta=0.9;emb=adam:0.19:b1=0.7;gate=muon:0.026:beta=0.7;up=muon:0.026:beta=0.7;head=adam:0.048:b1=0.3:b2=0.999;norm=adam:0.019:b2=0.9;q=muon:0.067:beta=0.8;k=muon:0.067:beta=0.8;v=muon:0.051:beta=0.7;o=muon:0.051:beta=0.7"  # the five-world row at width 48, a start only
GROUPS = dict(qk=("q", "k"), vo=("v", "o"), gu=("gate", "up"), down=("down",)); ADAMS = ("emb", "head", "norm"); MATS = ("q", "k", "v", "o", "gate", "up", "down")
TUNED = dict(vo=("v", "o"), qk=("q", "k"), ffn=("gate", "up", "down"))
SWAPS = dict(vo=("v", "o"), v=("v",), o=("o",), qk=("q", "k"), q=("q",), k=("k",), ffn=("gate", "up", "down"), gate=("gate",), up=("up",), down=("down",), qkffn=("q", "k", "gate", "up", "down"), all=MATS)
PATH = None


def work(a):
    a.device = f"cuda:{(mp.current_process()._identity[0] - 1) % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    return runq.job(a)


STEPS = 0


def job(s, tier, w): return env.parser().parse_args(["--spec", arms.show(s), "--tier", tier, "--world", str(w), "--evals", "tune", "--tag", "map", "--results", PATH, "--steps", str(STEPS)])


class Runner:
    def __init__(s, tier, par, hours): s.tier, s.end = tier, time.time() + hours * 3600; s.ex = cf.ProcessPoolExecutor(par, mp_context=mp.get_context("spawn")); s.rows = {}; s.load()

    def load(s):
        if os.path.exists(PATH):
            for l in open(PATH):
                r = json.loads(l)
                if r.get("tier") == s.tier: s.rows[(r["spec"], r["world"])] = r

    def need(s, specs, worlds):
        todo = {(arms.show(x), w): job(x, s.tier, w) for x in specs for w in worlds}; todo = {k: v for k, v in todo.items() if k not in s.rows}
        if not todo: return
        if time.time() > s.end: raise SystemExit("time budget used; rerun the same command to continue")
        fs = [s.ex.submit(work, a) for a in todo.values()]
        for f in cf.as_completed(fs):
            r = f.result()
            if r.get("error"): print("ERROR", r["spec"][:100], r["error"], flush=True)
        s.load()

    def val(s, x, w):
        r = s.rows.get((arms.show(x), w))
        return r["scores"]["text.tune.loss"] if r and r["finite"] and not r["opt"]["unstable"] else 99.

    def mean(s, x, worlds): return float(np.mean([s.val(x, w) for w in worlds]))


def scaled(b, parts, f): return {m: dict(g, lr=round(g["lr"] * f, 5)) if m in parts else g for m, g in b.items()}


def walk(R, make, grid, worlds, label):
    """Scores grid points, and keeps stepping past whichever end wins, at most twice each way."""
    grid = list(grid); ratio = grid[1] / grid[0]
    for _ in range(3):
        R.need([make(f) for f in grid], worlds); v = [R.mean(make(f), worlds) for f in grid]; i = int(np.argmin(v))
        if i == 0: grid.insert(0, grid[0] / ratio)
        elif i == len(grid) - 1: grid.append(grid[-1] * ratio)
        else: break
    v = {f: R.mean(make(f), worlds) for f in grid if all((arms.show(make(f)), w) in R.rows for w in worlds)}; best = min(v, key=v.get); print(f"  {label}: {best:g} {v[best]:.4f} of {len(v)}", flush=True); return best


def tune_base(R, worlds):
    b = arms.parse(MUON); muon = [m for m in b if b[m]["rule"] == "muon"]
    f = walk(R, lambda f: scaled(b, muon, f), (.5, 1, 2), worlds, "all Muon lrs x"); b = scaled(b, muon, f)
    f = walk(R, lambda f: scaled(b, ADAMS, f), (.5, 1, 2), worlds, "all Adam lrs x"); b = scaled(b, ADAMS, f)
    opts = [{m: dict(g, beta=x) if m in muon else g for m, g in b.items()} for x in (.8, .9, .95)]; R.need(opts, worlds); b = min(opts + [b], key=lambda x: R.mean(x, worlds))
    for g, parts in GROUPS.items():
        f = walk(R, lambda f: scaled(b, parts, f), (.63, 1, 1.6), worlds, f"{g} lr x"); b = scaled(b, parts, f)
    return b


def swapped(b, lrs): return {m: dict(rule="adam", lr=lrs[m]) if m in lrs else g for m, g in b.items()}


def split(b, heads, f): return {m: dict(rule="muonsplit", lr=round(g["lr"] * f, 5), beta=g["beta"], h=heads) if m in ("q", "k") else g for m, g in b.items()}


def main():
    global PATH, STEPS
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("--results", required=True); p.add_argument("--tune", default="0,1"); p.add_argument("--held", default="5,6,7,8,9"); p.add_argument("--par", type=int, default=4); p.add_argument("--hours", type=float, default=11); p.add_argument("--out", default=""); p.add_argument("--time-only", action="store_true"); a = p.parse_args()
    PATH = os.path.abspath(a.results); tune = [int(x) for x in a.tune.split(",")]; held = [int(x) for x in a.held.split(",")]; R = Runner(a.tier, a.par, a.hours); heads = env.TIERS[a.tier]["heads"]
    if a.time_only:
        STEPS = 200; t = time.time(); R.need([arms.parse(MUON)], [0, 1, 2, 5]); print(a.tier, "seconds for 4 runs at once, 200 steps", round(time.time() - t, 1), [(r["s_per_step"], r["train_s"], r["eval_s"]) for r in R.rows.values() if r["steps"] == 200], flush=True); return R.ex.shutdown()
    print(a.tier, "tuning Muon", flush=True); b = tune_base(R, tune); print("base", arms.show(b), f"{R.mean(b, tune):.4f}", flush=True)
    lrs = {}
    for name, parts in TUNED.items():  # one Adam lr per group, chosen on tune worlds; single matrices and joint moves inherit it
        lr = walk(R, lambda f: swapped(b, {m: f for m in parts}), (.004, .01, .025), tune, f"Adam on {name}, lr")
        for m in parts: lrs[m] = lr
    f = walk(R, lambda f: split(b, heads, f), (1, 2, 4), tune, "per-head q,k, lr x"); picks = {name: swapped(b, {m: lrs[m] for m in parts}) for name, parts in SWAPS.items()}; picks["split"] = split(b, heads, f)
    R.need([b] + list(picks.values()), held); B = np.array([R.val(b, w) for w in held]); out = dict(tier=a.tier, base=arms.show(b), base_tune=round(R.mean(b, tune), 4), base_held=round(float(B.mean()), 4), s_per_step=R.rows[(arms.show(b), held[0])]["s_per_step"], adam_lr=lrs, split_factor=f, gaps={})
    print(f"{a.tier} base held-out {B.mean():.4f}", flush=True)
    for name, s in picks.items():
        g = B - np.array([R.val(s, w) for w in held]); out["gaps"][name] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(g) ** .5), 4), wins=int((g > 0).sum()), worlds=np.round(g, 4).tolist(), spec=arms.show(s)); print(f"{name}: {g.mean():+.4f} error {g.std(ddof=1) / len(g) ** .5:.4f} wins {(g > 0).sum()}/{len(g)}", flush=True)
    if a.out: open(a.out, "a").write(json.dumps(out) + "\n")
    R.ex.shutdown()


if __name__ == "__main__":
    main()
