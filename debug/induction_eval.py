"""Re-measure saved final models (from induction.py --ckpt) with probes that do not depend on how well the model predicts rare tokens.
For each checkpoint, on n probe rows [EOT, r, r]:
  first, second: mean loss on the first and the second copy of r (the second copy alone is the copying measure; first minus second is the old ICL score)
  acc: share of second-copy tokens whose top prediction is the copied token
Probes: uniform (r drawn uniformly from the vocabulary, length 50, as in training probes), freq (r drawn by unigram frequency), off63 (uniform, length 63, so the repeat sits 63 tokens back, like the 64 of training repeats).
Also held-out loss on all tune rows of the world.
python3 debug/induction_eval.py --ckpt results/induction/ck_rep25x4000 --steps 4000 --out results/induction/eval_rep25x4000.json"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import env, evals
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import induction  # noqa: F401, registers the tiers


def rows(W, n, T, freq, seed=99):
    rng = np.random.default_rng(seed)
    if freq: p = W.counts[W.keep].astype(float); p /= p.sum(); r = rng.choice(len(W.keep), (n, T), p=p)
    else: r = rng.integers(0, W.V - 1, (n, T))
    return torch.as_tensor(np.concatenate([np.full((n, 1), W.look[env.EOT]), r, r], 1))


@torch.no_grad()
def probe(model, x, T, dev, bs=200):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(dev); lg = model(xb)[:, :-1]; tg = xb[:, 1:]
        nll = F.cross_entropy(lg.flatten(0, 1), tg.flatten(), reduction="none").view(tg.shape)
        hit = (lg.argmax(-1) == tg).float()
        out.append((nll[:, 1:T].mean(1), nll[:, T + 1:2 * T].mean(1), hit[:, T + 1:2 * T].mean(1)))
    f, s, a = (torch.cat(z).cpu().numpy() for z in zip(*out))
    return dict(first=round(float(f.mean()), 4), second=round(float(s.mean()), 4), icl=round(float((f - s).mean()), 4), acc=round(float(a.mean()), 4), second_se=round(float(s.std() / len(s) ** .5), 4))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--steps", type=int, default=4000); p.add_argument("--tier", default="w64x60k"); p.add_argument("--n", type=int, default=1000); p.add_argument("--out", required=True); p.add_argument("--only", default="")
    a = p.parse_args(); dev = os.environ.get("IND_DEV", "cpu"); c = dict(env.TIERS[a.tier]); c["own_shard"] = False; out = json.load(open(a.out)) if os.path.exists(a.out) else {}; Ws = {}
    for f in sorted(glob.glob(os.path.join(a.ckpt, f"*_{a.steps}*.pt"))):
        name, w, o = re.match(r"(.+)_w(\d+)_\d+(_o\d+)?\.pt", os.path.basename(f)).groups(); w = int(w); key = f"{name}/w{w}" + (o or "")
        if key in out or (a.only and not re.search(a.only, key)): continue
        z = torch.load(f, map_location=dev, weights_only=False)
        if z["step"] != a.steps: continue
        if w not in Ws: Ws[w] = env.World(c, 10, w)
        W = Ws[w]; model = env.LM(c, W.V).to(dev); model.load_state_dict(z["model"]); model.eval(); ctx = evals.Ctx(model, W, c, dev)
        r = dict(held=round(float(np.mean(ctx.row_loss(W.tune))), 4), n_held=len(W.tune))
        for k, (T, fq) in dict(uniform=(50, False), freq=(50, True), off63=(63, False)).items(): r[k] = probe(model, rows(W, a.n, T, fq), T, dev)
        out[key] = r; json.dump(out, open(a.out, "w"), indent=1)
        print(key, r["held"], *(f"{k}: 2nd {r[k]['second']:.3f} acc {r[k]['acc']:.3f} icl {r[k]['icl']:+.3f}" for k in ("uniform", "freq", "off63")), flush=True)


if __name__ == "__main__":
    main()
