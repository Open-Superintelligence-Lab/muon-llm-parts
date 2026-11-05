"""Quick screen of optimizers on one part, on tune rows of world 0. Everything else stays at the tuned --base arm, so a gap is the part's effect. Each name gets 5 lrs 1.6x apart, and a best lr at the edge moves the grid.

python3 debug/attn_screen.py 5s qk x_partial x_ada_muon muonsplit [--seconds 120]
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import arms, env, runq, sandbox, search


def main():
    p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("part"); p.add_argument("names", nargs="+"); p.add_argument("--results", default=search.RESULTS); p.add_argument("--seconds", type=float, default=120); p.add_argument("--base", default="muon"); a = p.parse_args()
    x = env.parser().parse_args([]); x.__dict__.update(tier=a.tier, results=a.results, tag="search", moe=0, tied=False, qknorm=True); book = search.Book(os.path.join(HERE, a.results)); budget = time.time() + a.seconds
    ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; base = arms.parse([r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == a.base and r["tier"] == "5s"][-1]["spec"])
    ref = search.try_all([base], x, book, budget)[0][0]; print(f"tuned {a.base} {ref:.4f}", flush=True); parts = a.part.split(",")
    for n in a.names:
        if n.startswith("x_"): sandbox.load(n)
        def make(f):
            s = {k: dict(g) for k, g in base.items()}
            for m in parts: s[m] = dict(rule=n, lr=search.round_lr(base[m]["lr"] * f), **({"beta": base[m]["beta"]} if not n.startswith("x_") and "beta" in base[m] else {}))  # a built-in rule takes the tuned momentum
            return s
        fs = [.4, .63, 1, 1.6, 2.5]; seen = {}
        for moves in range(3):
            new = [f for f in fs if f not in seen]; seen.update(zip(new, (v for v, _ in search.try_all([make(f) for f in new], x, book, budget))))
            b = min(seen, key=seen.get)
            if b == min(seen): fs = [round(b / 1.6, 3)]
            elif b == max(seen): fs = [round(b * 1.6, 3)]
            else: break
        print(f"{n}: tune {seen[b]:.4f}, gap to {a.base} {ref - seen[b]:+.4f}, lr x{b:g}, " + " ".join(f"{f:g}:{seen[f]:.3f}" for f in sorted(seen)), flush=True)
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
