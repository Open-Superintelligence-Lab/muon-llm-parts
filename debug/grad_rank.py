"""How concentrated each matrix's gradient is while the tuned arm trains: stable rank (squared size over squared top singular value) as a share of full rank, averaged over steps. A low share means a few directions dominate, which is where equalizing directions changes the step most.

python3 debug/grad_rank.py "<spec>"
"""
import os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn.functional as F
import arms, env, rules


def main():
    spec = arms.parse(sys.argv[1]); c = dict(env.TIERS["5s"]); out = {}
    for world in (5, 6, 7):
        W = env.World(c, c["steps"], world); torch.manual_seed(world); model = env.LM(c, W.V, True, False, 0); opt = rules.ModuleOpt(arms.groups(spec, list(model.named_parameters()))); X = torch.as_tensor(W.look[W.rows])
        for step in range(c["steps"]):
            idx = (step * c["batch"] + np.arange(c["batch"])) % len(X); x = X[idx]; F.cross_entropy(model(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten(), ignore_index=W.unk).backward()
            if step % 5 == 0:
                for n, w in model.named_parameters():
                    m = re.search(r"\.(q|k|v|o|gate|up|down)\.weight$", n)
                    if m and w.grad is not None and w.grad.norm() > 0:
                        S = torch.linalg.svdvals(w.grad); out.setdefault(m.group(1), []).append(float((S ** 2).sum() / S[0] ** 2 / len(S)))
            opt.step(min((step + 1) / 20, 1.0, (c["steps"] - step) / (.3 * c["steps"])), measure=False); model.zero_grad(set_to_none=True)
    for m in ("q", "k", "v", "o", "gate", "up", "down"): print(m, f"{np.mean(out[m]):.3f}")


if __name__ == "__main__":
    main()
