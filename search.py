"""Tune and search on tune rows only. Search rows hold no report score, so nothing here can read one.

python3 search.py tune muon sym --tier 5s [--from-tier 5s]    tune each recipe's lr per module
python3 search.py rule --tier 5s                              K1: per-module rule and lr, from the best tuned recipe
python3 search.py lronly --tier 5s                            K1 stricter control: same settings as the rule search, on lrs only
python3 search.py random --tier 5s --evals-total N            K1 check: random rule and lr per module with N settings
python3 search.py custom x_name --modules qk,vo --tier 5s      an invented optimizer from optimizers/x_name.py on the listed modules, starting from the best recipe

Every arm gets the same protocol and the same lr groups, one lr per module: passes over the modules with 5 lrs 2.5x apart, then 3 lrs 1.6x apart, then 3 lrs 1.25x apart.
A best lr at the edge of the first grid moves that grid, at most 3 times, so a choice ends inside its grid. A coordinate never accepts a value worse than the one it holds.
Then every arm tunes momentum per module (beta, or b1 and b2 for Adam), then lrs once more, because an invented optimizer can pick its own momentum and must not win on that alone.
Momentum and lr then move together, first on all the modules of one rule and then per module, because Muon and Adam fix their step size, so a lower momentum can want a much higher lr that one-at-a-time moves never reach. Momentum then gets one more pass.
The lr control for K1 is the best tuned recipe. It has had more settings than the rule search, because all the recipes were tuned first; both counts are saved.
The rule search anchors each rule's lrs on what tuning chose for that rule on that module, and it starts from the best recipe's scores. Its menu leaves out manifold rules (about 3x the time per step, over beat.py's cap), and search.py rule --manifold adds them.
"""
import argparse, json, math, os, random, sys, time
import arms, env, runq, sandbox

PASSES = ((5, 2.5), (3, 1.6), (3, 1.25)); RESULTS = "tune_results.jsonl"
MATRIX_RULES = ("adam", "muon", "rownorm", "polar", "hybrid", "lion", "sgdm", "normsgd")  # leftpolar equals polar on tall matrices and takes V x V eigendecompositions on embedding and head
BETAS = (.5, .7, .8, .9, .95, .98); JOINT_BETAS = (.3, .5, .7, .8, .9, .95); JOINT_LRS = (.63, 1, 1.6, 2.56, 4.1, 6.5); KNOBS = dict(adam=dict(b1=(.5, .7, .8, .9, .95), b2=(.9, .95, .99, .999))); DEFAULTS = dict(adam=dict(b1=.9, b2=.95), lion=dict(beta=.9), sgdm=dict(beta=.9))
NORM_RULES = ("adam", "norm1d", "sgdm", "lion"); MANIFOLD = ("mm_stiefel", "mm_dgram"); HIDDEN = ("qk", "vo", "gu", "down")


def spec_args(s, a, **k):
    x = env.parser().parse_args([]); x.__dict__.update(tier=a.tier, steps=a.steps, world=0, results=a.results, spec=arms.show(s), tag=a.tag, moe=a.moe, tied=a.tied, qknorm=a.qknorm, **k); return x


class Book:
    """Reads tune scores for finished settings. A setting that blew up, or that the instability rule flags, scores as infinite."""
    def __init__(s, path): s.path = path; s.seen = {}; s.load()
    def load(s):
        if os.path.exists(s.path):
            for l in open(s.path):
                r = json.loads(l); s.seen[r["key"]] = r
    def tune(s, a):
        r = s.seen.get(env.key_of(a, arms.parse(a.spec)))
        if r is None: return None
        v = r["scores"].get("text.tune.loss") if r["finite"] and not r["opt"]["unstable"] else None
        return v if v is not None else math.inf


def round_lr(x): return float(f"{x:.2g}")


def grid(c, n, f): return [round_lr(c * f ** (i - n // 2)) for i in range(n)]


def try_all(specs, a, book, budget):
    """Runs the specs and returns (tune score, spec) pairs. Stops the program if the time budget runs out, and a rerun continues."""
    js = [spec_args(s, a) for s in specs]; left = runq.run(js, seconds=max(1, budget - time.time()))
    book.load()
    if left or time.time() > budget: print(f"time budget used, {left} settings left in this batch: rerun the same command", flush=True); sys.exit(3)
    return [(book.tune(j), s) for j, s in zip(js, specs)]


def with_lr(cur, m, lr):
    s = {k: dict(g) for k, g in cur.items()}; s[m]["lr"] = lr; return s


def with_rule(cur, m, r, lr):
    s = {k: dict(g) for k, g in cur.items()}; s[m] = dict(rule=r, lr=lr); return s


def mods(a): return [m for m in arms.MODULES + (arms.MOE if a.moe else ()) if not (m == "head" and a.tied)]


def tune_lrs(spec, a, book, budget, count, passes=PASSES):
    """The protocol every arm gets. Returns the best tune score and spec, and it never moves to a worse setting."""
    best = try_all([spec], a, book, budget)[0][0]; count.add(arms.show(spec))
    for n, f in passes:
        for m in mods(a):
            lrs = grid(spec[m]["lr"], n, f); moves = 3 if n == 5 else 0
            while lrs:
                res = try_all([with_lr(spec, m, x) for x in lrs], a, book, budget); count.update(arms.show(with_lr(spec, m, x)) for x in lrs); v, s = min(res, key=lambda p: p[0])
                if v < best: best, spec = v, s
                now = spec[m]["lr"]; edge = now == max(lrs) and 1 or now == min(lrs) and -1  # a best at the edge moves the grid one lr further
                lrs = [round_lr(now * f ** edge)] if edge and moves else []; moves -= 1
            print(f"  {f}x lr of {m}: tune {best:.4f}, settings {len(count)}", flush=True)
    return best, spec


def momentum_pass(spec, a, book, budget, count):
    """Momentum per module, then one more lr pass. An invented rule keeps its own momentum, which is part of its design."""
    best = try_all([spec], a, book, budget)[0][0]
    for m in mods(a):
        r = spec[m]["rule"]
        for knob, vals in ({} if r.startswith("x_") else KNOBS.get(r, dict(beta=BETAS))).items():
            now = spec[m].get(knob, DEFAULTS.get(r, {}).get(knob, .95)); cands = [{k: dict(g, **({knob: v} if k == m else {})) for k, g in spec.items()} for v in vals if v != now]
            res = try_all(cands, a, book, budget); count.update(arms.show(s) for s in cands); v, s = min(res, key=lambda p: p[0])
            if v < best: best, spec = v, s
        print(f"  momentum of {m}: tune {best:.4f}, settings {len(count)}", flush=True)
    return tune_lrs(spec, a, book, budget, count, PASSES[1:2])


def knob_of(rule): return None if rule.startswith("x_") else "b1" if rule == "adam" else "beta"


def tune_joint(spec, a, book, budget, count):
    """Moves momentum and lr together, first for every module with the same rule family, then for each module with the neighbouring momentums. An invented rule keeps its own momentum and gets the lr moves."""
    best = try_all([spec], a, book, budget)[0][0]; family = {}
    for m in mods(a): family.setdefault("adam" if spec[m]["rule"] == "adam" else "x" if spec[m]["rule"].startswith("x_") else "muon" if spec[m]["rule"].startswith("muon") else spec[m]["rule"], []).append(m)
    for fam, ms in family.items():
        kn = knob_of(spec[ms[0]]["rule"]); cands = []
        for v in JOINT_BETAS if kn and fam in ("adam", "muon") else (None,):
            for f in JOINT_LRS:
                s = {k: dict(g) for k, g in spec.items()}
                for m in ms:
                    s[m]["lr"] = round_lr(s[m]["lr"] * f)
                    if v is not None: s[m][kn] = v
                cands.append(s)
        res = try_all(cands, a, book, budget); count.update(arms.show(s) for s in cands); v, s = min(res, key=lambda p: p[0])
        if v < best: best, spec = v, s
        print(f"  momentum and lr of all {fam}: tune {best:.4f}, settings {len(count)}", flush=True)
    for m in mods(a):
        kn = knob_of(spec[m]["rule"])
        if not kn: continue
        now = spec[m].get(kn, DEFAULTS.get(spec[m]["rule"], {}).get(kn, .95)); near = sorted(JOINT_BETAS + (.98,), key=lambda x: abs(x - now))[:3]; cands = []
        for v in near:
            for f in (.63, 1, 1.6):
                cands.append({k: dict(g, **(dict(lr=round_lr(g["lr"] * f), **{kn: v}) if k == m else {})) for k, g in spec.items()})
        res = try_all(cands, a, book, budget); count.update(arms.show(s) for s in cands); v, s = min(res, key=lambda p: p[0])
        if v < best: best, spec = v, s
        print(f"  momentum and lr of {m}: tune {best:.4f}, settings {len(count)}", flush=True)
    return best, spec


def tune_momentum(spec, a, book, budget, count):
    """The momentum part of the protocol every arm gets."""
    best, spec = momentum_pass(spec, a, book, budget, count); best, spec = tune_joint(spec, a, book, budget, count)
    return momentum_pass(spec, a, book, budget, count)


def save(d, a):
    d["joint"] = True
    with open(os.path.join(env.HERE, a.choices), "a") as f: f.write(json.dumps(d) + "\n")
    print("chosen:", json.dumps(d), flush=True)


def choices(a):
    p = os.path.join(env.HERE, a.choices); return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def tuned(a, tier=None):
    return {r["arm"]: r for r in choices(a) if r["kind"] == "tune" and r.get("joint") and r["tier"] == (tier or a.tier) and r["moe"] == a.moe and r.get("tied", False) == a.tied and r.get("qknorm", True) == a.qknorm}


def best_recipe(a):
    t = tuned(a)
    if not t: raise SystemExit("tune the recipes first")
    k = min(t, key=lambda x: t[x]["tune"]); return k, arms.parse(t[k]["spec"])


def anchors(a):
    """rule -> module -> the lr that tuning chose for that rule on that module, the same rule on any module as a fallback through LRS."""
    out = {}
    for r in tuned(a).values():
        for m, g in arms.parse(r["spec"]).items():
            out.setdefault(g["rule"], {}).setdefault(m, g["lr"])
    return out


def rule_search(a, book, budget):
    """Per-module rule and lr, from the best recipe. Anchored rules get 3 lrs 2.5x apart, unanchored rules 5. The current setting stays a candidate, so a move never loses."""
    name, spec = best_recipe(a); an = anchors(a); count = set(); best = try_all([spec], a, book, budget)[0][0]
    for m in mods(a):
        cands = []; menu = NORM_RULES if m == "norm" else MATRIX_RULES + (MANIFOLD if a.manifold and m in HIDDEN else ()) + (("routernorm",) if m == "router" else ())
        for r in menu:
            lr = an.get(r, {}).get(m); lrs = grid(lr, 3, 2.5) if lr else grid(arms.LRS[r], 5, 2.5)
            cands += [with_rule(spec, m, r, x) for x in lrs]
        res = try_all(cands, a, book, budget); count.update(arms.show(s) for s in cands); v, s = min(res, key=lambda p: p[0])
        if v < best: best, spec = v, s
        print(f"  rule of {m}: tune {best:.4f}, now {spec[m]['rule']}:{spec[m]['lr']:g}, settings {len(count)}", flush=True)
    best, spec = tune_lrs(spec, a, book, budget, count, PASSES[1:]); best, spec = tune_momentum(spec, a, book, budget, count)
    save(dict(kind="rule", start=name, manifold=a.manifold, spec=arms.show(spec), tune=best, evals=len(count), tier=a.tier, moe=a.moe), a)


def lr_only(a, book, budget):
    """The stricter control for K1: the best recipe with about as many extra settings as the rule search, all spent on lrs, no rule changes."""
    name, spec = best_recipe(a); count = set(); best, spec = tune_lrs(spec, a, book, budget, count, ((7, 1.6), (7, 1.25), (7, 1.12), (7, 1.06)))
    save(dict(kind="lronly", start=name, manifold=False, spec=arms.show(spec), tune=best, evals=len(count), tier=a.tier, moe=a.moe), a)


def custom(a, book, budget, name):
    """An invented rule gets the protocol every recipe gets: same passes, same lr groups, on the best recipe with the listed modules swapped."""
    _, spec = best_recipe(a); use = a.modules.split(",")
    if not name.startswith("x_") or not all(m in spec for m in use): raise SystemExit("custom needs x_name and --modules from " + ",".join(mods(a)))
    sandbox.load(name)
    for m in use: spec[m] = dict(rule=name, lr=arms.LRS.get(name, spec[m]["lr"]))
    count = set(); best, spec = tune_lrs(spec, a, book, budget, count); best, spec = tune_momentum(spec, a, book, budget, count)
    save(dict(kind="custom", arm=name, modules=use, code=sandbox.code_hash(name), spec=arms.show(spec), tune=best, evals=len(count), tier=a.tier, moe=a.moe), a)


def random_search(a, book, budget, total):
    r = random.Random(0); m = mods(a); _, base = best_recipe(a); specs = [base]
    for _ in range(total - 1):
        s = {k: dict(g) for k, g in base.items()}
        for k in m:
            x = r.choice(NORM_RULES if k == "norm" else MATRIX_RULES); s[k] = dict(rule=x, lr=round_lr(arms.LRS[x] * 2.5 ** r.uniform(-2, 2)))
        specs.append(s)
    res = try_all(specs, a, book, budget); v, s = min(res, key=lambda p: p[0])
    save(dict(kind="random", spec=arms.show(s), tune=v, evals=len(res), tier=a.tier, moe=a.moe), a)


def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["tune", "rule", "lronly", "random", "custom"]); p.add_argument("names", nargs="*"); p.add_argument("--tier", default="5s"); p.add_argument("--modules", default=""); p.add_argument("--from-tier", default=""); p.add_argument("--steps", type=int, default=0); p.add_argument("--manifold", action="store_true")
    p.add_argument("--moe", type=int, default=0); p.add_argument("--tied", action="store_true"); p.add_argument("--no-qknorm", dest="qknorm", action="store_false"); p.add_argument("--seconds", type=float, default=240); p.add_argument("--evals-total", type=int, default=0)
    p.add_argument("--results", default=RESULTS); p.add_argument("--choices", default="choices.jsonl"); p.add_argument("--tag", default="search"); a = p.parse_args()
    book = Book(os.path.join(env.HERE, a.results)); budget = time.time() + a.seconds
    done = {(r["kind"], r.get("arm"), r["tier"], r["moe"], r.get("tied", False), r.get("qknorm", True), r.get("manifold", False), r.get("joint", False)) for r in choices(a)}
    old = {r["arm"]: r for r in choices(a) if r["kind"] == "tune" and r["tier"] == a.tier and r["moe"] == a.moe and r.get("tied", False) == a.tied and r.get("qknorm", True) == a.qknorm}  # lrs tuned under the same protocol, which momentum tuning continues from
    if a.mode == "tune":
        start = tuned(a, a.from_tier) if a.from_tier else {}
        for n in a.names:
            if ("tune", n, a.tier, a.moe, a.tied, a.qknorm, False, True) in done: continue
            print(n, flush=True); count = set(); before = old[n]["evals"] if n in old else 0
            if n in old: spec = arms.parse(old[n]["spec"])
            else: spec = arms.parse(start[n]["spec"]) if n in start else arms.parse(n); _, spec = tune_lrs(spec, a, book, budget, count)
            best, spec = tune_momentum(spec, a, book, budget, count)
            save(dict(kind="tune", arm=n, spec=arms.show(spec), tune=best, evals=before + len(count), tier=a.tier, moe=a.moe, tied=a.tied, qknorm=a.qknorm), a)
    elif a.mode == "rule":
        if ("rule", None, a.tier, a.moe, False, True, a.manifold, True) not in done: rule_search(a, book, budget)
    elif a.mode == "lronly":
        if ("lronly", None, a.tier, a.moe, False, True, False, True) not in done: lr_only(a, book, budget)
    elif a.mode == "custom":
        for n in a.names: custom(a, book, budget, n)
    elif ("random", None, a.tier, a.moe, False, True, False, True) not in done: random_search(a, book, budget, a.evals_total or 150)
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
