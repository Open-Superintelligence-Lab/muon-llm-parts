"""Random optimizer search, one part at a time. Each candidate takes one random choice per stage of optimizers/x_rand.py, with every shape tried equally often. Everything is chosen on worlds 0,1 and scored on worlds 5-9, tune rows, against the tuned baseline. Finished runs are skipped, so the same command continues a stopped search. Positive gain beats the baseline.

python3 debug/random_opt.py --n 100 --results results/random/w48.jsonl --out results/random/w48_out.json [--par 8] [--hours .13] [--parts vo,qk] [--check]
--single tries Muon with one stage changed (debug/random_opt.py --single --tier 5sx4 ...); part "all" is every hidden matrix.
"""
import argparse, json, os, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms

PARTS = dict(all=("q", "k", "v", "o", "gate", "up", "down"), vo=("v", "o"), qk=("q", "k"), ffn=("gate", "up", "down"), emb=("emb",), head=("head",)); LR0 = dict(emb=.32, head=.02)  # RowNorm's tuned lr has the same step size as these rules
P3 = {2: (-.25, -.1, .1, .25, .5, .75), 11: (.5, .75, .9), 12: (.25, .5, .75), 13: (0, .25, .5), 15: (.1, .25, .5), 16: (.1, .25, .5), 17: (.1, .25, .5), 18: (0, .34, .67), 19: (.25, .5, 1), 20: (.25, .5, .75), 21: (.25, .5, 1), 22: (.1, .25, .5), 7: (.1, .5, 1)}
P4 = {1: (.25, .5, 1), 2: (.5, 1), 3: (-.5, .5, 1), 4: (-.5, .5, 1), 5: (0, .25, 1), 6: (0, .25, 1), 9: (.25, .5)}; P5 = {1: (0, .25, 1), 3: (.01, .1, 1), 4: (.01, .1, .5), 6: (.1, .5)}
NAMES = dict(pre="none centre-rows centre-cols global-norm row-norm col-norm sign-mix adam-scale clip row-clip grad-diff".split(), mom="nesterov heavy none two-speed quasi-hyperbolic pid restart sign-agree row-restart after-shape".split(), shape="muon raw soft-power rownorm colnorm sinkhorn sign tanh qr head-muon col-block-muon spectrum-clip low-rank drop-top log-spectrum muon+sign muon+raw muon+adam few-ns rank-transform top-entries muon-keep-rows muon-times-size row-then-muon col-then-muon".split(), size="const grad-size agreement weight-size row-size fade grow tangent row-tangent cycle".split(), post="none lookahead keep-norm decay ortho-pull keep-rows centre".split())
sig = lambda x: float(f"{x:.4g}")


def sample(i, part):
    """Candidate i for this part. i alone sets it, so more candidates extend the same list."""
    r = np.random.default_rng(zlib.crc32(f"{part}{i}".encode())); pick = lambda x: float(x[int(r.integers(len(x)))]); order = np.random.default_rng(zlib.crc32(part.encode())).permutation(25)
    while True:
        k = dict(shape=int(order[i % 25]), beta=pick((.7, .8, .9, .95)))
        for name, top in (("pre", 10), ("mom", 9), ("size", 9), ("post", 6)):
            if r.random() < .4: k[name] = int(r.integers(1, top + 1))
        if k.get("mom", 0) == 2: k.pop("beta")
        if k.get("pre", 0) in (6, 7, 8, 9, 10): k["p1"] = pick((.25, .5, 1))
        if k.get("mom", 0) in (3, 4, 5, 7): k["p2"] = pick((.25, .5, .75))
        if k["shape"] in P3: k["p3"] = pick(P3[k["shape"]])
        if k.get("size", 0) in P4: k["p4"] = pick(P4[k["size"]])
        if k.get("post", 0) in P5: k["p5"] = pick(P5[k["post"]])
        if (k.get("pre", 0) in (7, 10)) + (k.get("mom", 0) != 2) + (k.get("mom", 0) in (3, 5)) + (k.get("post", 0) == 1) <= 3: return k


def single():
    """Muon with exactly one stage changed, at Muon's tuned momentum .8. The list order is fixed."""
    P1 = {6: (.25, .5, 1), 7: (.25, .5, 1), 8: (.25, .5, 1), 9: (.25, .5, 1), 10: (.25, .5, 1)}; P2 = {3: (.25, .5, .75), 4: (.25, .5, .75), 5: (.25, .5, .75), 7: (.25, .5, .75)}
    C = [dict(beta=x) for x in (.6, .9, .95)]
    for st, top, P, key in (("pre", 10, P1, "p1"), ("mom", 9, P2, "p2"), ("shape", 24, P3, "p3"), ("size", 9, P4, "p4"), ("post", 6, P5, "p5")):
        for n in range(1, top + 1): C += [dict({st: n, "beta": .8}, **({key: x} if n in P else {})) for x in P.get(n, (None,))]
    return C


def label(k): return " + ".join(NAMES[s][int(k.get(s, 0))] for s in NAMES if k.get(s, 0) or s == "shape") + "".join(f" {x}={k[x]}" for x in ("beta", "p1", "p2", "p3", "p4", "p5") if x in k)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=100); p.add_argument("--results", required=True); p.add_argument("--out", required=True); p.add_argument("--par", type=int, default=8); p.add_argument("--hours", type=float, default=.13); p.add_argument("--parts", default=""); p.add_argument("--top2", type=int, default=20); p.add_argument("--top3", type=int, default=6); p.add_argument("--tier", default="5s"); p.add_argument("--long", default=""); p.add_argument("--long-tier", default="5sx4"); p.add_argument("--long-min", type=float, default=.002); p.add_argument("--check", action="store_true"); p.add_argument("--single", action="store_true"); a = p.parse_args()
    mp.PATH = os.path.abspath(a.results); os.makedirs(os.path.dirname(mp.PATH), exist_ok=True); parts = a.parts.split(",") if a.parts else list(PARTS)
    last = lambda f: arms.parse([json.loads(l) for l in open(f) if "gaps" in l][-1]["base"]); B250, B1k = last("results/add/w48_out.jsonl"), last("results/add/w48x1000_out.jsonl"); tune, held = [0, 1], [5, 6, 7, 8, 9]; H = 4
    def make(b, part, k, f): return {m: dict(k, rule="x_rand", lr=sig(LR0.get(part, g["lr"]) * f), h=H) if m in PARTS[part] else g for m, g in b.items()}
    R = mp.Runner(a.tier, a.par, a.hours)
    if a.tier == "5sx4": B250 = B1k
    def need(R, specs, worlds):
        specs = list(specs)
        for i in range(0, len(specs), 48): R.need(specs[i:i + 48], worlds)
    if a.check:
        same = {"beta": .8}; need(R, [B250] + [make(B250, x, same, 1) for x in ("vo", "qk", "ffn")], [0]); print("check, all stages 0 against Muon, world 0:", R.val(B250, 0), [R.val(make(B250, x, same, 1), 0) for x in ("vo", "qk", "ffn")], flush=True); return R.ex.shutdown()
    need(R, [B250], tune + held); base = {w: R.val(B250, w) for w in tune + held}; print(f"{a.tier} base tune {np.mean([base[w] for w in tune]):.4f} held {np.mean([base[w] for w in held]):.4f}", flush=True); out = dict(n=a.n, base=arms.show(B250), screen={}, tune={}, held={}, long={})
    if a.long:
        # survivors from a finished search, at a longer run: retune lr on worlds 0,1 from the longer baseline, then score on worlds 5-9. --long takes the search's out file; --long-tier debug only.
        done = json.load(open(a.long)); R2 = mp.Runner(a.long_tier, a.par, a.hours); need(R2, [B1k], tune + held); base2 = {w: R2.val(B1k, w) for w in tune + held}; print(f"{a.long_tier} base tune {np.mean([base2[w] for w in tune]):.4f} held {np.mean([base2[w] for w in held]):.4f}", flush=True); long = {}
        for part, rows in done["held"].items():
            long[part] = []
            for r in rows:
                if r["mean"] < a.long_min: continue
                opts = [make(B1k, part, r["k"], f) for f in (.25, .35, .5, .7, 1, 1.4, 2, 2.8, 4)]; need(R2, opts, tune); s = min(opts, key=lambda x: float(np.mean([R2.val(x, w) for w in tune]))); need(R2, [s], held); g = np.array([base2[w] - R2.val(s, w) for w in held])
                long[part].append(dict(i=r["i"], name=r["name"], k=r["k"], spec=arms.show(s), held250=r["mean"], mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / 5 ** .5), 4), wins=int((g > 0).sum()))); print(f"{part} {r['name']}: 250 steps {r['mean']:+.4f}, {a.long_tier} {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True); json.dump(long, open(a.out, "w"), indent=0)
        R2.ex.shutdown(); R.ex.shutdown(); return
    G = (.35, 1, 2.8); gain = lambda s, ws: float(np.mean([base[w] - R.val(s, w) for w in ws]))
    for part in parts:
        C = single() if a.single else [sample(i, part) for i in range(a.n)]; a.n = len(C); need(R, [make(B250, part, k, f) for k in C for f in G], [0]); rows = []
        for i, k in enumerate(C):
            f = max(G, key=lambda f: gain(make(B250, part, k, f), [0])); rows.append(dict(i=i, k=k, f=f, gain=gain(make(B250, part, k, f), [0]), name=label(k)))
        rows.sort(key=lambda r: -r["gain"]); out["screen"][part] = [dict(r, gain=round(r["gain"], 4)) for r in rows]; print(f"== {part}: {a.n} candidates, {sum(r['gain'] > 0 for r in rows)} beat the base on world 0, best {rows[0]['gain']:+.4f}, median {rows[len(rows) // 2]['gain']:+.4f}", flush=True)
        S = rows[:a.top2]; opts = {r["i"]: [make(B250, part, r["k"], r["f"] * x) for x in (.7, 1, 1.4)] for r in S}; need(R, [x for v in opts.values() for x in v], tune)
        for r in S: r["s"] = max(opts[r["i"]], key=lambda x: gain(x, tune)); r["tune"] = gain(r["s"], tune); r["lr_f"] = r["s"]
        S.sort(key=lambda r: -r["tune"]); out["tune"][part] = [dict(i=r["i"], name=r["name"], tune=round(r["tune"], 4)) for r in S]; T = S[:a.top3]; need(R, [r["s"] for r in T], held); out["held"][part] = []
        for r in T:
            g = np.array([base[w] - R.val(r["s"], w) for w in held]); out["held"][part].append(dict(i=r["i"], name=r["name"], k=r["k"], spec=arms.show(r["s"]), tune=round(r["tune"], 4), mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / 5 ** .5), 4), wins=int((g > 0).sum()))); print(f"  {r['name']}: tune {r['tune']:+.4f}, held {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5", flush=True)
        json.dump(out, open(a.out, "w"), indent=0)
    R.ex.shutdown()


if __name__ == "__main__":
    main()
