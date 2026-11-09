"""Weight decay on query and key only, under Muon with QK-norm (transplant of the block-output-norm + weight decay result).

Runs env.py's trainer with one change: the Muon rule honours a `wd` knob (decoupled, w *= 1 - lr * wd) and records each
matrix's Frobenius norm at the first and last step, so the run reports how much each group grew.

    python3 debug/qk_wd.py --spec muon --tier 5s --world 5                       # baseline
    python3 debug/qk_wd.py --spec muon --set qk=muon:.02:wd=.1 --tier 5s --world 5  # decay on query and key only
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, rules, env, arms

NORMS = {}  # id(w) -> [first norm, last norm, module]
_muon = rules.RULES["muon"]


MODULE = {}  # id(w) -> module name, filled when the optimizer is built
_init = rules.ModuleOpt.__init__


def init(s, groups):
    for m, g in groups.items():
        for w in g["params"]: MODULE[id(w)] = m
    _init(s, groups)


def muon_wd(w, g, st, lr, k):
    n = NORMS.setdefault(id(w), [float(w.norm()), 0.0, MODULE.get(id(w), "?")])
    wd = k.get("wd", 0.0)
    if wd: w.mul_(1 - lr * wd)
    _muon(w, g, st, lr, k); n[1] = float(w.norm())


rules.RULES["muon"] = muon_wd; rules.ModuleOpt.__init__ = init

if __name__ == "__main__":
    a = env.parser().parse_args(); spec = arms.parse(a.spec)
    for s in a.set: spec.update(arms.parse(s))
    a.results = os.environ.get("QKWD_RESULTS", "results/qk_wd.jsonl"); os.makedirs("results", exist_ok=True)
    r = env.train(a, spec)
    grow = collections.defaultdict(list)
    for first, last, m in NORMS.values(): grow[m].append(last / first)
    print(arms.show(spec))
    print(f"  tune loss {r['scores'].get('text.tune.loss')}  web loss {r['scores'].get('text.web.loss')}  train {r['train_s']}s")
    print("  norm growth (last/first): " + ", ".join(f"{m} {sum(v) / len(v):.2f}x" for m, v in sorted(grow.items())))
