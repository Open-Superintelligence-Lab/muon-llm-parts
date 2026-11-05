"""Gives q, k, v, o, gate and up their own optimizer settings for this run only, without changing arms.py, on tune rows. The baseline is the tuned muon-split arm, whose q and k share one setting, as do v and o, and gate and up.

Each change is matrix=rule:lr-multiplier[:knob=value...], and + joins changes in one run. A rule left empty keeps the matrix's rule.
python3 debug/fine_parts.py v=muonsplit:1:h=4 q=:1.6 k=:.63+q=:1.6 --worlds 0 --results path
"""
import argparse, json, os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
import arms, env, runq

FINE = dict(q="qk", k="qk", v="vo", o="vo", gate="gu", up="gu"); old = arms.module_of


def module_of(name, w):
    m = re.search(r"\.(q|k|v|o|gate|up)(\.\d+)?\.weight$", name)
    return m.group(1) if m else old(name, w)


arms.module_of = module_of  # workers import this file again when they start, so they get this too


def base_spec(arm):
    ch = [json.loads(l) for l in open(os.path.join(HERE, "choices.jsonl"))]; b = arms.parse([r for r in ch if r.get("joint") and r["kind"] == "tune" and r["arm"] == arm and r["tier"] == "5s"][-1]["spec"]); out = {m: g for m, g in b.items() if m not in ("qk", "vo", "gu")}
    for m, p in FINE.items(): out[m] = dict(b[p])
    return out


def change(base, text):
    s = {m: dict(g) for m, g in base.items()}
    for one in text.split("+"):
        m, _, rest = one.partition("="); bits = rest.split(":"); g = dict(s[m]); g["lr"] = round(g["lr"] * float(bits[1]), 5)
        if bits[0]:
            if bits[0] != g["rule"] and g["rule"] == "muonsplit": g.pop("h", None)
            g["rule"] = bits[0]
        for b in bits[2:]: g[b.split("=")[0]] = float(b.split("=")[1])
        s[m] = g
    return s


def main():
    p = argparse.ArgumentParser(); p.add_argument("changes", nargs="*"); p.add_argument("--arm", default="muon-split"); p.add_argument("--worlds", default="0"); p.add_argument("--results", required=True); p.add_argument("--seconds", type=float, default=200); p.add_argument("--tier", default="5s"); p.add_argument("--steps", type=int, default=0); a = p.parse_args()
    base = base_spec(a.arm); worlds = [int(w) for w in a.worlds.split(",")]; specs = {c: change(base, c) for c in a.changes}
    job = lambda s, w: env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--world", str(w), "--evals", "tune", "--tag", "fine", "--results", a.results, "--steps", str(a.steps)])
    left = runq.run([job(s, w) for s in [base] + list(specs.values()) for w in worlds], a.seconds)
    if runq.EX: runq.EX.shutdown()
    if left: print("left", left, "rerun the same command")
    rows = {}
    for l in open(a.results):
        r = json.loads(l); rows[(r["spec"], r["world"], r.get("steps"))] = r
    pick = lambda s, w: [r for k, r in rows.items() if k[0] == arms.show(s) and k[1] == w and (not a.steps or k[2] == a.steps)]
    val = lambda s, w: (lambda r: r[-1]["scores"]["text.tune.loss"] if r and r[-1]["finite"] and not r[-1]["opt"]["unstable"] else float("nan"))(pick(s, w))
    print("base", [round(val(base, w), 4) for w in worlds], flush=True)
    for c, s in specs.items():
        g = np.array([val(base, w) - val(s, w) for w in worlds])
        print(f"{c}: {g.mean():+.4f}" + (f" error {g.std(ddof=1) / len(g) ** .5:.4f} wins {(g > 0).sum()}/{len(g)} {np.round(g, 4).tolist()}" if len(g) > 1 else ""), flush=True)


if __name__ == "__main__":
    main()
