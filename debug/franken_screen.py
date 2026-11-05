"""Tries one piece of optimizers/x_franken.py at a time on Muon's matrices, on tune rows of world 0. Everything else stays at the tuned --base arm. The first line runs every piece off, which must equal the base exactly.

python3 debug/franken_screen.py qk,vo,gu,down graft=1 lion=.3 --seconds 200
A name with commas joins pieces: graft=1,warm=25.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import arms, env, runq, sandbox, search


def main():
    p = argparse.ArgumentParser(); p.add_argument("parts"); p.add_argument("pieces", nargs="*"); p.add_argument("--base", default="muon-split"); p.add_argument("--seconds", type=float, default=200); p.add_argument("--tier", default="5s"); p.add_argument("--rule", default="x_franken"); a = p.parse_args()
    x = env.parser().parse_args([]); x.__dict__.update(tier=a.tier, results=search.RESULTS, tag="search", moe=0, tied=False, qknorm=True); book = search.Book(os.path.join(HERE, x.results)); budget = time.time() + a.seconds
    ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; base = arms.parse([r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == a.base and r["tier"] == "5s"][-1]["spec"]); parts = a.parts.split(","); sandbox.load(a.rule)

    def make(f, kn):
        s = {m: dict(g) for m, g in base.items()}
        for m in parts: s[m] = dict(dict(rule=a.rule, lr=search.round_lr(base[m]["lr"] * f), beta=base[m].get("beta", .95), h=4 if base[m]["rule"] == "muonsplit" else 1), **kn)
        return s

    ref = search.try_all([base], x, book, budget)[0][0]; off = search.try_all([make(1, {})], x, book, budget)[0][0]; print(f"{a.base} {ref:.4f}, all pieces off {off:.4f}", flush=True); assert a.rule not in ("x_franken", "x_muon_shape") or abs(ref - off) < 1e-3, "all off must equal the base"
    for name in a.pieces:
        kn = {t.split("=")[0]: float(t.split("=")[1]) for t in name.split(",")}; fs = [.63, 1, 1.6]; seen = {}
        for moves in range(3):
            new = [f for f in fs if f not in seen]; seen.update(zip(new, (v for v, _ in search.try_all([make(f, kn) for f in new], x, book, budget))))
            b = min(seen, key=seen.get)
            if b == min(seen): fs = [round(b / 1.6, 3)]
            elif b == max(seen): fs = [round(b * 1.6, 3)]
            else: break
        print(f"{name}: tune {seen[b]:.4f}, gap {ref - seen[b]:+.4f}, lr x{b:g}, " + " ".join(f"{f:g}:{seen[f]:.3f}" for f in sorted(seen)), flush=True)
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
