"""Where does the copied token come from? For saved final models, on n frequency-matched probe rows [EOT, r, r] (the rows induction_eval.py uses),
at every second-copy position:
  att[l][h]: attention to the token after the earlier copy (the induction target), on these same rows
  dla[l][h]: direct logit credit of head h in layer l for the copied token: its output, scaled by the final RMSNorm at that position,
             through the unembedding, as copied-token logit minus mean logit over the vocabulary (Elhage et al. 2021)
  mlp[l]: the same for each layer's MLP; emb: for the token embedding
  by_freq: first- and second-copy loss split by how common the copied token is (rank among kept tokens: <100, 100-999, >=1000)
python3 debug/induction_heads.py --ckpt results/induction/ck_repvar0.5 --out results/induction/heads_repvar50x4000.json"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import induction  # noqa: F401
from induction_eval import rows


@torch.no_grad()
def credit(model, x, T, rank):
    B, t = x.shape; dev = x.device; pos = torch.arange(T + 1, 2 * T, device=dev); tgt = x[:, pos + 1]  # predicting x[pos+1] from pos
    H = model.head.weight; Hm = H.mean(0); parts = {}
    def score(c):  # c: B, t, d contribution to the residual; returns mean over rows and positions of copied-token logit minus mean logit
        z = c[:, pos] * scale
        return float(((z * H[tgt]).sum(-1) - z @ Hm).mean())
    e = model.emb(x); comps = [("emb", e)]; atts = []
    for l, b in enumerate(model.blocks):
        h = b.n1(e); q, k, v = (m(h).view(B, t, b.h, b.dk).transpose(1, 2) for m in (b.q, b.k, b.v))
        s = (env.rope(b.qn(q)) @ env.rope(b.kn(k)).transpose(-1, -2)) / b.dk ** .5
        p = s.masked_fill(torch.ones(t, t, dtype=torch.bool, device=dev).triu(1), float("-inf")).softmax(-1); y = p @ v  # B, heads, t, dk
        atts.append([float(p[:, hh, pos, pos - T + 1].mean()) for hh in range(b.h)])
        Wo = b.o.weight; heads = [y[:, hh] @ Wo[:, hh * b.dk:(hh + 1) * b.dk].T for hh in range(b.h)]
        after = b(e); attn = sum(heads); comps += [(f"L{l}H{hh}", heads[hh]) for hh in range(b.h)]; comps.append((f"mlp{l}", after - e - attn)); e = after
    scale = model.norm.weight / e.pow(2).mean(-1, keepdim=True).add(model.norm.eps or 1e-6).sqrt()  # RMSNorm at each position, frozen
    scale = scale[:, pos]
    for n, c in comps: parts[n] = round(score(c), 3)
    logits = model.head(model.norm(e)); nll = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(B, -1)
    first, second = nll[:, 1:T], nll[:, T + 1:2 * T]; rk = rank[x[:, 2:T + 1]]  # nll[:, i] predicts x[:, i+1]; first copy targets x[2..T], second x[T+2..2T]
    by = {k: dict(first=round(float(first[m].mean()), 3), second=round(float(second[m].mean()), 3), n=int(m.sum())) for k, m in (("top100", rk < 100), ("100-999", (rk >= 100) & (rk < 1000)), ("1000+", rk >= 1000))}
    return atts, parts, by


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--steps", type=int, default=4000); p.add_argument("--tier", default="w64x60k"); p.add_argument("--n", type=int, default=500); p.add_argument("--out", required=True); p.add_argument("--only", default="")
    a = p.parse_args(); dev = os.environ.get("IND_DEV", "cpu"); c = dict(env.TIERS[a.tier]); c["own_shard"] = False; out = json.load(open(a.out)) if os.path.exists(a.out) else {}; Ws = {}
    for f in sorted(glob.glob(os.path.join(a.ckpt, f"*_{a.steps}.pt"))):
        name, w = re.match(r"(.+)_w(\d+)_\d+\.pt", os.path.basename(f)).groups(); w = int(w); key = f"{name}/w{w}"
        if key in out or (a.only and not re.search(a.only, key)): continue
        z = torch.load(f, map_location=dev, weights_only=False)
        if z["step"] != a.steps: continue
        if w not in Ws: Ws[w] = env.World(c, 10, w)
        W = Ws[w]; model = env.LM(c, W.V).to(dev); model.load_state_dict(z["model"]); model.eval()
        rank = torch.full((W.V,), W.V, dtype=torch.long); order = np.argsort(-W.counts[W.keep]); rank[torch.as_tensor(order)] = torch.arange(len(order)); rank = rank.to(dev)
        x = rows(W, a.n, 50, True).to(dev); atts, parts = None, {}; bys = []
        for i in range(0, a.n, 100):
            at, pa, by = credit(model, x[i:i + 100], 50, rank); bys.append(by)
            atts = at if atts is None else [[u + v for u, v in zip(r1, r2)] for r1, r2 in zip(atts, at)]; parts = {k: parts.get(k, 0) + v for k, v in pa.items()}
        k = a.n // 100; atts = [[round(v / k, 3) for v in r] for r in atts]; parts = {n: round(v / k, 3) for n, v in parts.items()}
        by = {b: {m: round(float(np.mean([y[b][m] for y in bys])), 3) for m in ("first", "second")} for b in bys[0]}
        heads = {n: v for n, v in parts.items() if n.startswith("L")}; top = max(heads, key=heads.get)
        out[key] = dict(att=atts, dla=parts, by_freq=by, top_head=top); json.dump(out, open(a.out, "w"), indent=1)
        print(key, "top head", top, heads[top], "att", atts[int(top[1])][int(top[3])], "heads sum", round(sum(heads.values()), 2), "mlp sum", round(sum(v for n, v in parts.items() if n.startswith("mlp")), 2), "emb", parts["emb"], "| 2nd by freq", {b: by[b]["second"] for b in by}, flush=True)


if __name__ == "__main__":
    main()
