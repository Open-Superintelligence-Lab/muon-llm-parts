"""Why do q,k not need Muon? Trains the tuned Muon baseline, all Adam, and Adam plus Muon on one group (the add rows of results/add/), then measures, on the report rows of each world:
  cut: loss rise when one group's trained weights keep only their top fraction f of directions (whole matrix, or per head for q,k,v,o)
  rarity: loss by target-token frequency (evals.rarity)
One training run per setting and world; the analysis itself trains nothing.

python3 debug/mech.py --tier 5sx4 --add results/add/w48x1000_out.jsonl --out results/mech/w48x1000.json
python3 debug/mech.py --tier w64 --map kaggle/out/map-a/map_out.jsonl --out results/mech/w64_map.json  (Muon base and v,o on Adam at a Kaggle map width)
"""
import argparse, copy, json, multiprocessing as mp, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import arms, env, evals
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import map_parts  # noqa: F401, gives q, k, v, o, gate and up their own settings

GROUPS = dict(qk=("q", "k"), vo=("v", "o"), ffn=("gate", "up", "down"))
FRACS = (.125, .25, .5, .75)


def cut(w, f, heads, by_head, col):
    """Keep the top fraction f of singular directions of w, per head when by_head (rows of q,k,v, columns of o)."""
    if not by_head:
        k = max(1, round(f * min(w.shape))); U, S, V = torch.linalg.svd(w, full_matrices=False); return (U[:, :k] * S[:k]) @ V[:k]
    parts = w.chunk(heads, 1 if col else 0); out = []
    for p in parts:
        k = max(1, round(f * min(p.shape))); U, S, V = torch.linalg.svd(p, full_matrices=False); out.append((U[:, :k] * S[:k]) @ V[:k])
    return torch.cat(out, 1 if col else 0)


def loss(ctx): return float(np.mean(ctx.row_loss(ctx.W.val)))


def job(x):
    name, spec, tier, w, analyse, noqk, steps = x; torch.set_num_threads(1); held = []; LM = env.LM
    if steps: env.TIERS[tier] = dict(env.TIERS[tier], steps=steps)

    def hold(*a, **k):
        held.append(LM(*a, **k)); return held[-1]
    env.LM = hold
    a = env.parser().parse_args(["--tier", tier] + (["--no-qknorm"] if noqk else [])); a.world = w; a.evals = "tune"; a.threads = 1
    W = env.World(dict(env.TIERS[tier]), env.TIERS[tier]["steps"], w); env.train(a, spec, W=W, out=os.path.join(tempfile.mkdtemp(), "x.jsonl"))
    model = held[-1].eval(); c = env.TIERS[tier]; ctx = evals.Ctx(model, W, c, "cpu"); r = dict(name=name, world=w, loss=loss(ctx))
    r["rarity"] = {k: float(np.nanmean(v)) for k, v in evals.rarity(ctx).items()}
    if analyse:
        r["cut"] = {}; base = r["loss"]
        for g, mods in GROUPS.items():
            for by_head in ((False, True) if g != "ffn" else (False,)):
                for f in FRACS:
                    keep = {}
                    with torch.no_grad():
                        for b in model.blocks:
                            for m in mods:
                                lin = getattr(b, m); keep[(id(b), m)] = lin.weight.data.clone(); lin.weight.data = cut(lin.weight.data, f, c["heads"], by_head, m == "o")
                    r["cut"][f"{g}{'_head' if by_head else ''}_{f}"] = loss(ctx) - base
                    with torch.no_grad():
                        for b in model.blocks:
                            for m in mods: getattr(b, m).weight.data = keep[(id(b), m)]
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", required=True); p.add_argument("--add", default=""); p.add_argument("--out", required=True); p.add_argument("--worlds", default="5,6,7,8,9"); p.add_argument("--par", type=int, default=9); p.add_argument("--cut-only", action="store_true"); p.add_argument("--no-qknorm", action="store_true"); p.add_argument("--map", default="", help="a map_out.jsonl: trains that tier's Muon base and the base with v,o moved to tuned Adam, and cuts both"); p.add_argument("--steps", type=int, default=0, help="debug only: override the tier's steps"); a = p.parse_args()
    if a.map:
        row = [x for x in map(json.loads, open(a.map)) if x["tier"] == a.tier][-1]; specs = dict(muon=arms.parse(row["base"]), vo_adam=arms.parse(row["gaps"]["vo"]["spec"]))
    else:
        L = [json.loads(l) for l in open(a.add)]; row = [x for x in L if x.get("kind") == "add"][-1]
        specs = dict(muon=arms.parse(row["base"]))
        if not a.cut_only: specs.update(adam=arms.parse(row["adam"]), adam_muon_vo=arms.parse(row["rows"]["vo"]["spec"]), adam_muon_qk=arms.parse(row["rows"]["qk"]["spec"]))
    W = [int(x) for x in a.worlds.split(",")]; jobs = [(n, s, a.tier, w, True, a.no_qknorm, a.steps) for n, s in specs.items() for w in W]
    with mp.get_context("spawn").Pool(min(a.par, len(jobs))) as pool: res = pool.map(job, jobs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(dict(tier=a.tier, qknorm=not a.no_qknorm, worlds=W, specs={n: arms.show(s) for n, s in specs.items()}, runs=res), open(a.out, "w"), indent=0)
    by = lambda n: [r for r in res if r["name"] == n]
    for n in specs: print(n, "loss", round(np.mean([r["loss"] for r in by(n)]), 4))
    for n in specs:
        if not by(n): continue
        print(f"\n{n}: loss rise when a group keeps its top fraction of directions (mean over worlds, error)")
        for key in by(n)[0]["cut"]:
            v = np.array([r["cut"][key] for r in by(n)]); print(f"  {key:16s} {v.mean():+.4f} ({v.std(ddof=1) / len(v) ** .5 if len(v) > 1 else 0:.4f})")
    if by("adam"):
        print("\nrarity: gain over all Adam (positive = lower loss than Adam)")
        for n in specs:
            if n == "adam": continue
            print(" ", n, {k.split(".")[1]: round(float(np.mean([ra["rarity"][k] - rn["rarity"][k] for ra, rn in zip(by("adam"), by(n))])), 4) for k in by(n)[0]["rarity"]})


if __name__ == "__main__":
    main()
