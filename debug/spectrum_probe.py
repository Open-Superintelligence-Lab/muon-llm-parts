"""Do small singular directions of the update carry signal? One run with every hidden matrix on optimizers/x_hyc.py (exact polar unless the spec says otherwise). Every 50 steps it takes the SVD of each matrix's Nesterov momentum d = U S V^T, then reads the raw gradient G that arrives 1 and 20 steps later along each direction: z_i = u_i^T G v_i / |G| * sqrt(rows*cols). Pure noise gives z near 0 with spread 1; z > 0 means the gradient still pushes the same way, so a bigger step there would have lowered the loss. Prints, per part and per quarter of the spectrum (top singular values first), the mean z, and each quarter's share of |d|^2.

python3 debug/spectrum_probe.py "SPEC" [--tier 5sx4] [--world 0] [--every 50]
"""
import argparse, collections, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "debug"))
import map_parts  # noqa: F401, gives q, k, v, o, gate and up their own settings
import numpy as np, torch
import arms, env, sandbox

p = argparse.ArgumentParser(); p.add_argument("spec"); p.add_argument("--tier", default="5sx4"); p.add_argument("--world", type=int, default=0); p.add_argument("--every", type=int, default=50); a0 = p.parse_args()
NAME = {}; COUNT = collections.Counter(); OPEN = collections.defaultdict(list); Z = collections.defaultdict(list); SHARE = collections.defaultdict(list); LAGS = (1, 20); PH = ('0-300', '300-600', '600+')
real_groups = arms.groups


def groups(spec, named):
    for n, w in named: NAME[w.data_ptr()] = arms.module_of(n, w)
    return real_groups(spec, named)


arms.groups = groups; real_wrap = sandbox.wrap


def wrap(fn):
    inner = real_wrap(fn)

    def rule(w, g, st, lr, k):
        key = w.data_ptr(); t = COUNT[key] = COUNT[key] + 1; part = NAME.get(key, "?"); G = g.detach().float()
        for (t0, U, V) in [x for x in OPEN[key] if t - x[0] in LAGS]:
            z = ((U.T @ G) * V.T).sum(0) / (G.norm() + 1e-12) * (G.shape[0] * G.shape[1]) ** .5; Z[(part, f"{t - t0:>2}, steps {PH[min(2, t0 // 300)]}")].append(z.numpy())
        OPEN[key] = [x for x in OPEN[key] if t - x[0] < max(LAGS)]
        inner(w, g, st, lr, k)
        if t % a0.every == 0 and "b" in st:
            d = (G + k.get("beta", .95) * st["b"]).float(); U, S, Vh = torch.linalg.svd(d, full_matrices=False); OPEN[key].append((t, U, Vh.T)); SHARE[part].append((S ** 2 / (S ** 2).sum()).numpy())
    return rule


sandbox.wrap = wrap
a = env.parser().parse_args(["--spec", a0.spec, "--tier", a0.tier, "--world", str(a0.world), "--results", os.path.join(os.environ.get("TMPDIR", "/tmp"), "spectrum_probe.jsonl"), "--force"])
env.train(a, env.full_spec(a))
part_of = dict(q="qk", k="qk", v="vo", o="vo", gate="ffn", up="ffn", down="ffn")
rows = collections.defaultdict(dict)
for (m, lag), zs in Z.items():
    q = np.array_split(np.mean(zs, 0), 4); rows[(part_of.get(m, m), lag)].setdefault("z", []).append([float(x.mean()) for x in q])
for m, sh in SHARE.items():
    q = np.array_split(np.mean(sh, 0), 4); rows[(part_of.get(m, m), "share")].setdefault("z", []).append([float(x.sum()) for x in q])
for (part, lag), r in sorted(rows.items(), key=lambda x: (x[0][0], str(x[0][1]))):
    v = np.mean(r["z"], 0); print(f"{part} {'share of |d|^2' if lag == 'share' else f'z at lag {lag}'}: " + "  ".join(f"{x:+.2f}" if lag != "share" else f"{x:.2f}" for x in v), flush=True)
