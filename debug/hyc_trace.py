"""Where does the learned exponent go? Trains one run with optimizers/x_hyc.py on one part (the rest as the spec says) and prints each matrix's exponent c every 100 steps. Reads state only; the run itself is the normal env.train.

python3 debug/hyc_trace.py "SPEC" [--tier 5sx4] [--world 0]
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "debug"))
import map_parts  # noqa: F401, gives q, k, v, o, gate and up their own settings
import env, sandbox

p = argparse.ArgumentParser(); p.add_argument("spec"); p.add_argument("--tier", default="5sx4"); p.add_argument("--world", type=int, default=0); a0 = p.parse_args()
T = {}; N = [0]
real = sandbox.wrap


def wrap(fn):
    inner = real(fn)

    def rule(w, g, st, lr, k):
        inner(w, g, st, lr, k)
        if "c" in st: T.setdefault(id(st), []).append(round(st["c"], 3))
    return rule


sandbox.wrap = wrap
a = env.parser().parse_args(["--spec", a0.spec, "--tier", a0.tier, "--world", str(a0.world), "--results", os.path.join(os.environ.get("TMPDIR", "/tmp"), "hyc_trace.jsonl"), "--force"])
env.train(a, env.full_spec(a))
for i, v in enumerate(T.values()): print(f"matrix {i}: c every 100 steps", v[::100] + [v[-1]])
