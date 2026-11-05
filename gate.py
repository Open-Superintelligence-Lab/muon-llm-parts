"""The only way to score a design on fresh worlds. Nothing else reads these worlds, and their numbers stay outside the repo.

A person runs these. An AI agent gets search.py and optimizers/, and never gets gate.py, env.py, a shell or the seal folder.
python3 gate.py --seal                  once: draws the fresh worlds and locks the code
python3 gate.py --relock                after a person changes the code; the worlds stay
python3 gate.py rule --tier 5s          submits the last rule search of that tier
python3 gate.py x_name --tier 5s        submits the last tuned invented optimizer of that name

The design must come from choices.jsonl with a tune row behind it. The opponent is the best tuned recipe or lr-only control with at least 2/3 of the design's settings, so more tuning cannot win.
A design wins only if: the gap has the same sign in all 5 worlds; the mean gap clears max(.005, t x error) with t corrected for every design ever submitted; it stays stable; tokens and gradients match; time per step is at most 1.25x and state at most 2x the opponent's.
"""
import argparse, hashlib, json, os, random
import numpy as np
from scipy import stats

SEAL = os.environ.get("SEAL_DIR", os.path.expanduser("~/.muon-llm-parts")); os.environ["RUNS_DIR"] = os.path.join(SEAL, "runs")
import arms, env, runq, sandbox

LOCKED = ("env.py", "evals.py", "rules.py", "arms.py", "search.py", "runq.py", "sandbox.py", "gate.py"); FRESH = (15, 16, 17, 18, 19, 21, 22, 23, 24); N = 5  # worlds 0-4, 7, 10-14 and 20 were used in the old repo or the baseline table
MIN_GAP, COMPUTE, STATE, ALPHA, SCORE, FAIR = .005, 1.25, 2.0, .05, "text.web.loss", 1.5
RESULTS = os.path.join(SEAL, "results.jsonl"); LOG = os.path.join(SEAL, "log.jsonl"); SEALED = os.path.join(SEAL, "sealed.json")


def hashes(): return {f: hashlib.sha256(open(os.path.join(env.HERE, f), "rb").read()).hexdigest()[:16] for f in LOCKED}


def seal(relock):
    os.makedirs(SEAL, exist_ok=True); old = json.load(open(SEALED)) if os.path.exists(SEALED) else None
    if old and not relock: raise SystemExit("already sealed; --relock takes new code hashes and keeps the worlds")
    worlds = old["worlds"] if old else sorted(random.SystemRandom().sample(FRESH, N)); json.dump(dict(worlds=worlds, code=hashes()), open(SEALED, "w")); os.chmod(SEALED, 0o600); print("sealed", N, "worlds; code locked")


def log_rows(): return [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []


def choices(path=None): return [json.loads(l) for l in open(path or os.path.join(env.HERE, "choices.jsonl"))]


def refuse(name, tier, sealed, ch, tune_keys, now=None):
    """Returns (reason, design, opponent). Each check here is one way to cheat, and debug/cheats.py tries them all."""
    now = now or hashes(); changed = [f for f in LOCKED if now.get(f) != sealed["code"].get(f)]
    if changed: return f"locked code changed: {changed}", None, None
    kind = "rule" if name == "rule" else "custom"; d = [r for r in ch if r["kind"] == kind and r.get("joint") and r["tier"] == tier and r.get("moe", 0) == 0 and (kind == "rule" or r["arm"] == name)]
    if not d: return f"no {name} design at tier {tier} in choices.jsonl", None, None
    d = d[-1]; spec = arms.parse(d["spec"])
    for g in spec.values():
        if g["rule"].startswith("x_"):
            why = sandbox.check(open(sandbox.path_of(g["rule"])).read())
            if why: return f"{g['rule']} is refused: {why}", None, None
    a = env.parser().parse_args([]); a.__dict__.update(tier=tier, world=0, evals="tune")
    if env.key_of(a, spec) not in tune_keys: return "no tune row for this exact setting and optimizer file", None, None
    fair = [r for r in ch if r["kind"] in ("tune", "lronly") and r.get("joint") and r["tier"] == tier and r.get("moe", 0) == 0 and d["evals"] <= FAIR * r["evals"]]
    if not fair: return f"the design had {d['evals']} settings and no opponent was tuned with at least 2/3 of that", None, None
    return None, d, min(fair, key=lambda r: r["tune"])


def job(spec, tier, world, results): return env.parser().parse_args(["--spec", spec, "--tier", tier, "--world", str(world), "--evals", "full", "--tag", "gate", "--results", results])


def run(specs, tier, worlds, results, seconds):
    left = runq.run([job(s, tier, w, results) for s in specs for w in worlds], seconds)
    if runq.EX: runq.EX.shutdown(); runq.EX = None
    return left


def judge(base, design, tier, worlds, results, n):
    """Paired gap on report rows, base minus design, so positive means the design is better."""
    rows = {}
    for l in open(results):
        r = json.loads(l); rows[(r["spec"], r["tier"], r["world"])] = r
    gaps, why, rate = [], [], []; b, d = arms.show(arms.parse(base)), arms.show(arms.parse(design))
    for w in worlds:
        x, y = rows.get((b, tier, w)), rows.get((d, tier, w))
        if not x or not y: why.append("a world did not finish"); continue
        if not y["finite"] or y["opt"]["unstable"]: why.append(f"unstable: {y['opt'].get('why')}"); continue
        if y["tokens"] != x["tokens"] or y["grads_per_step"] != 1: why.append("tokens or gradients differ")
        if y["state_bytes"] > STATE * x["state_bytes"]: why.append(f"state {y['state_bytes']} is over {STATE}x {x['state_bytes']}")
        rate.append((x["s_per_step"], y["s_per_step"])); p, q = np.load(os.path.join(env.RUNS, x["key"] + ".npz")), np.load(os.path.join(env.RUNS, y["key"] + ".npz")); g = p[SCORE] - q[SCORE]; gaps.append((float(g.mean()), float(g.std(ddof=1) / len(g) ** .5)))
    out = dict(gaps=[round(g[0], 4) for g in gaps])
    if len(gaps) == len(worlds):
        v = np.array([g[0] for g in gaps]); se = max(v.std(ddof=1) / len(v) ** .5, float(np.mean([g[1] for g in gaps]))); need = max(MIN_GAP, float(stats.t.ppf(1 - ALPHA / 2 / n, len(v) - 1)) * se)
        out.update(mean=round(float(v.mean()), 4), error=round(se, 4), need=round(need, 4))
        if v.min() <= 0: why.append("the gap is not positive in every world")
        if v.mean() <= need: why.append(f"mean gap {v.mean():.4f} is not above {need:.4f}")
        a, c = min(r[0] for r in rate), min(r[1] for r in rate)
        if c > COMPUTE * a: why.append(f"time per step {c:.4f} is over {COMPUTE}x {a:.4f}")
    out.update(win=not why, why=why); return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("name", nargs="?"); p.add_argument("--tier", default="5s"); p.add_argument("--seal", action="store_true"); p.add_argument("--relock", action="store_true"); p.add_argument("--seconds", type=float, default=240); a = p.parse_args()
    if a.seal or a.relock: return seal(a.relock)
    if not os.path.exists(SEALED): raise SystemExit("run --seal first")
    s = json.load(open(SEALED)); keys = env.done_keys(os.path.join(env.HERE, "tune_results.jsonl")); why, d, o = refuse(a.name, a.tier, s, choices(), keys)
    if why: raise SystemExit("refused: " + why)
    seen = log_rows(); old = [r for r in seen if r["design"] == d["spec"] and r["tier"] == a.tier and r["base"] == o["spec"] and "verdict" in r]
    if old: print("already judged:", json.dumps(old[-1]["verdict"])); return
    if not any(r["design"] == d["spec"] and r["tier"] == a.tier for r in seen):
        with open(LOG, "a") as f: f.write(json.dumps(dict(design=d["spec"], tier=a.tier, base=o["spec"], name=a.name)) + "\n")  # counted before any score exists, so a bad result still counts
    n = len({(r["design"], r["tier"]) for r in log_rows()}); left = run([o["spec"], d["spec"]], a.tier, s["worlds"], RESULTS, a.seconds)
    if left: raise SystemExit(f"{left} runs left: rerun the same command")
    v = judge(o["spec"], d["spec"], a.tier, s["worlds"], RESULTS, n); v.update(opponent=o.get("arm") or o["kind"], settings=dict(design=d["evals"], opponent=o["evals"]), designs_submitted=n)
    with open(LOG, "a") as f: f.write(json.dumps(dict(design=d["spec"], tier=a.tier, base=o["spec"], name=a.name, verdict=v)) + "\n")
    print(json.dumps(v, indent=1)); print("WIN" if v["win"] else "NO WIN")


if __name__ == "__main__":
    main()
