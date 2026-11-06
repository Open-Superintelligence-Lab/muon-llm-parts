"""Does Muon make the induction head form earlier than Adam, and which part does it? Trains each arm with env.train's loop and schedule,
and every --every steps records the tune loss (64 rows) and two induction measures on 64 rows of [EOT, r, r] with r = 20 random tokens:
  icl: loss on the first copy of r minus loss on the second copy (Olsson et al. 2022), in nats
  att: per block and head, mean attention from each second-copy token to the token after its earlier copy
One training run per arm and world.

python3 debug/induction.py --tier 5sx4 --add results/add/w48x1000_out.jsonl --out results/induction/w48x1000.json
"""
import argparse, json, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import arms, env, evals, rules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import map_parts  # noqa: F401, gives q, k, v, o, gate and up their own settings

T = 20  # repeated span length; jobs with longer rows use up to 50
for L, st in ((128, 2000), (128, 4000)): env.TIERS[f"iL{L}x{st}"] = dict(env.TIERS["5s"], L=L, steps=st)  # longer rows, where repeated spans exist
for st in (60000,): env.TIERS[f"w64x{st // 1000}k"] = dict(map_parts.SHAPE, width=64, layers=4, heads=4, steps=st)  # Kaggle: the map's width-64 model, read long enough for an induction head to form
for V in (8000,): env.TIERS[f"iV{V}L128x2000"] = dict(env.TIERS["5s"], V=V, L=128, steps=2000)  # rare words (names) kept, which is what copying is for


def probe_rows(W, n=64):
    rng = np.random.default_rng(1234); r = rng.integers(0, W.V - 1, (n, T)); eot = W.look[env.EOT]  # W.V - 1 is the unknown token, never drawn
    return torch.as_tensor(np.concatenate([np.full((n, 1), eot), r, r], 1))


@torch.no_grad()
def induction(model, x):
    """icl score and per-head induction attention for rows x = [EOT, r, r]."""
    e = model.emb(x); att = []
    for b in model.blocks:
        B, t, d = e.shape; h = b.n1(e); q, k = (m(h).view(B, t, b.h, b.dk).transpose(1, 2) for m in (b.q, b.k))
        s = (env.rope(b.qn(q)) @ env.rope(b.kn(k)).transpose(-1, -2)) / b.dk ** .5
        p = s.masked_fill(torch.ones(t, t, dtype=torch.bool, device=s.device).triu(1), float("-inf")).softmax(-1)  # B, heads, t, t
        qi = torch.arange(T + 2, 2 * T + 1, device=s.device); kj = qi - T + 1  # second-copy r_j sits at T+1+j; the token after its first copy sits at j+2
        att.append([round(float(p[:, hh, qi, kj].mean()), 4) for hh in range(b.h)]); e = b(e)
    logits = model.head(model.norm(e)) if not model.tied else model.norm(e) @ model.emb.weight.T
    nll = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(x.shape[0], -1)  # nll[:, i] predicts x[:, i+1]
    first, second = nll[:, 1:T].mean(), nll[:, T + 1:2 * T].mean()
    return round(float(first - second), 4), att


def job(x):
    global T
    name, spec, tier, w, steps, every, stream, deadline, rep, ckdir, repvar = x; torch.set_num_threads(1); torch.manual_seed(w); c = dict(env.TIERS[tier]); steps = steps or c["steps"]; T = min(50, (c["L"] - 1) // 2)
    dev = f"cuda:{(mp.current_process()._identity[0] - 1) % torch.cuda.device_count()}" if torch.cuda.is_available() else os.environ.get("IND_DEV", "cpu")
    if stream: c["own_shard"] = False  # the world only supplies vocabulary and probe rows; training rows come from --stream
    W = env.World(c, 10 if stream else steps, w); model = env.LM(c, W.V).to(dev); n_named = list(model.named_parameters())
    opt = rules.ModuleOpt(arms.groups(spec, n_named)); ctx = evals.Ctx(model, W, c, dev); tune = W.tune[:64]; P = probe_rows(W).to(dev); look = torch.as_tensor(W.look)
    if stream:  # long runs: read the listed shards in order, one row after another, instead of the world's 100M-token file
        shards = [np.memmap(f, dtype=np.uint16, mode="r", offset=1024) for f in stream.split(",")]; per = [len(z) // c["L"] for z in shards]
        def batch(step):
            r = step * c["batch"] + np.arange(c["batch"]); out = []
            for i in r:
                j = 0
                while i >= per[j]: i -= per[j]; j += 1
                out.append(np.asarray(shards[j][i * c["L"]:(i + 1) * c["L"]]))
            xb = look[torch.as_tensor(np.stack(out).astype(np.int64))]
            if rep:  # the first rep share of each batch repeats its first half, so copying pays off early; the same rows in every arm
                k = int(round(rep * len(xb))); h = c["L"] // 2
                if repvar:  # content copying: per row, a 20-token span reappears 21-87 tokens later, at a place drawn from the step, so no fixed look-back distance works
                    g = np.random.default_rng(10_000_019 + step)
                    for i in range(k):
                        a0 = int(g.integers(1, c["L"] - 2 * 20 - 1)); b0 = int(g.integers(a0 + 21, c["L"] - 20 + 1)); xb[i, b0:b0 + 20] = xb[i, a0:a0 + 20]
                else: xb[:k, h:2 * h] = xb[:k, :h]
            return xb
        if steps * c["batch"] > sum(per): raise SystemExit(f"{steps} steps need {steps * c['batch'] * c['L']} tokens, the shards have {sum(per) * c['L']}")
    else:
        X = torch.as_tensor(W.look[W.rows]); batch = lambda step: X[(step * c["batch"] + np.arange(c["batch"])) % len(X)]
    curve, start, used, t0 = [], 0, 0.0, time.time(); ck = ckdir and os.path.join(ckdir, f"{name}_w{w}_{steps}.pt")
    if ck and os.path.exists(ck):  # resume: weights, optimizer state in parameter order, curve so far; batches depend only on the step
        z = torch.load(ck, map_location=dev, weights_only=False); model.load_state_dict(z["model"]); curve, start, used = z["curve"], z["step"], z["secs"]
        for (_, q), st in zip(n_named, z["opt"]): opt.state[id(q)] = st
    late = False
    for step in range(start, steps + 1):
        late = bool(deadline) and time.time() > deadline
        if (step % every == 0 or step == steps or late) and not (curve and curve[-1]["step"] == step):
            model.eval(); icl, att = induction(model, P); curve.append(dict(step=step, loss=round(float(np.mean(ctx.row_loss(tune))), 4), icl=icl, att=att, secs=round(used + time.time() - t0))); model.train()
            if dev != "cpu": print(name, w, curve[-1]["step"], curve[-1]["loss"], icl, max(max(h) for h in att), flush=True)
        if step == steps or late: break
        sc = min((step + 1) / 20, 1.0, (steps - step) / (0.3 * steps))  # env.train's schedule
        xb = batch(step).to(dev)
        loss = F.cross_entropy(model(xb)[:, :-1].flatten(0, 1), xb[:, 1:].flatten(), ignore_index=W.unk); loss.backward()
        opt.step(sc); model.zero_grad(set_to_none=True)
    if ck: os.makedirs(ckdir, exist_ok=True); torch.save(dict(model=model.state_dict(), opt=[opt.state[id(q)] for _, q in n_named], curve=curve, step=step, secs=used + time.time() - t0), ck)
    return dict(name=name, world=w, secs=round(used + time.time() - t0, 1), stopped_early=late, step=step, curve=curve)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5sx4"); p.add_argument("--add", default="results/add/w48x1000_out.jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--worlds", default="5,6,7,8,9"); p.add_argument("--arms", default="adam,muon,adam_muon_vo,adam_muon_qk"); p.add_argument("--steps", type=int, default=0); p.add_argument("--every", type=int, default=10); p.add_argument("--par", type=int, default=9)
    p.add_argument("--map", default="", help="a map_out.jsonl: arms muon, adam (the map's all-Adam row), muon_vo_adam, muon_qk_adam, muon_ffn_adam"); p.add_argument("--stream", default=""); p.add_argument("--max-min", type=float, default=0); p.add_argument("--rep", type=float, default=0); p.add_argument("--repvar", action="store_true", help="--rep rows copy a 20-token span to a random later place instead of repeating the first half"); p.add_argument("--extra", default="", help="json file {arm name: spec string} for arms the map does not have"); p.add_argument("--ckpt", default="", help="folder: save each arm when --max-min runs out and resume it on the next call"); a = p.parse_args()
    if a.map:
        row = [x for x in map(json.loads, open(a.map)) if x["tier"] == a.tier.split("x")[0]][-1]; g = row["gaps"]
        specs = dict(muon=arms.parse(row["base"]), adam=arms.parse(g["all"]["spec"]), muon_vo_adam=arms.parse(g["vo"]["spec"]), muon_qk_adam=arms.parse(g["qk"]["spec"]), muon_ffn_adam=arms.parse(g["ffn"]["spec"]))
        specs.update({f"gap_{k}": arms.parse(v["spec"]) for k, v in g.items()})  # every map row by name, e.g. gap_q, gap_qkffn, gap_split
    else:
        row = [x for x in map(json.loads, open(a.add)) if x.get("kind") == "add"][-1]
        specs = dict(muon=arms.parse(row["base"]), adam=arms.parse(row["adam"]), adam_muon_vo=arms.parse(row["rows"]["vo"]["spec"]), adam_muon_qk=arms.parse(row["rows"]["qk"]["spec"]))
    if a.extra: specs.update({n: arms.parse(v) for n, v in json.load(open(a.extra)).items()})
    specs = {n: specs[n] for n in a.arms.split(",")}; W = [int(x) for x in a.worlds.split(",")]
    jobs = [(n, s, a.tier, w, a.steps, a.every, a.stream, a.max_min and time.time() + 60 * a.max_min, a.rep, a.ckpt, a.repvar) for n, s in specs.items() for w in W]
    with mp.get_context("spawn").Pool(min(a.par, len(jobs))) as pool: res = pool.map(job, jobs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(tier=a.tier, rep=a.rep, repvar=a.repvar, stream=a.stream, steps=a.steps or env.TIERS[a.tier]["steps"], worlds=W, specs={n: arms.show(s) for n, s in specs.items()}, runs=res), open(a.out, "w"))
    for n in specs:
        rs = [r for r in res if r["name"] == n]; last = [r["curve"][-1] for r in rs]
        print(f"{n:14s} loss {np.mean([l['loss'] for l in last]):.4f} icl {np.mean([l['icl'] for l in last]):+.3f} best head att {np.mean([max(max(h) for h in l['att']) for l in last]):.3f} at step {rs[0]['step']} ({rs[0]['secs']}s)")


if __name__ == "__main__":
    main()
