"""Reads each group's weight matrices after one training run per setting, to see how the step shape changes the weights themselves. Diagnostic only, one run per setting on world 0.

python3 debug/weights.py 5sx4 results/add/w48x1000_out.jsonl results/spec/w48x1000_out.jsonl results/add/w48x1000_out.jsonl results/spec/weights_w48x1000.json
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_parts as mp
import arms, env
import torch

tier, base_file, spec_file, add_file, out = sys.argv[1:6]; base = arms.parse([json.loads(l) for l in open(base_file) if "gaps" in l][-1]["base"]); rows = [json.loads(l) for l in open(spec_file)][-1]["rows"]; adam = [json.loads(l) for l in open(add_file) if '"kind": "add"' in l][-1]["adam"]
SETTINGS = dict(muon=base, grad_all=arms.parse(rows["grad_all"]["spec"]), grad_vo=arms.parse(rows["grad_vo"]["spec"]), grad_qk=arms.parse(rows["grad_qk"]["spec"]), adam=arms.parse(adam)); held = []; LM = env.LM


def hold(*x, **k):
    held.append(LM(*x, **k)); return held[-1]


env.LM = hold; res = {}
def stats(model):
    d = {}
    for n, w in model.named_parameters():
        m = arms.module_of(n, w)
        if m not in mp.MATS or w.ndim != 2: continue
        S = torch.linalg.svdvals(w.detach()); d.setdefault(m, []).append((float(S.pow(2).sum() / S[0] ** 2), float(S[len(S) // 2] / S[0]), float(S[-1] / S[0]), float(S[0])))
    return {m: dict(stable_rank=round(sum(x[0] for x in v) / len(v), 2), median_ratio=round(sum(x[1] for x in v) / len(v), 3), min_ratio=round(sum(x[2] for x in v) / len(v), 4), top=round(sum(x[3] for x in v) / len(v), 3)) for m, v in d.items()}


for name, spec in SETTINGS.items():
    a = env.parser().parse_args(["--tier", tier]); a.world = 0; a.evals = "tune"; a.threads = int(os.environ.get("OMP_NUM_THREADS", 2)); r = env.train(a, spec, out=os.path.join(tempfile.mkdtemp(), "x.jsonl")); res[name] = dict(loss=r["scores"]["text.tune.loss"], parts=stats(held[-1])); print(name, res[name]["loss"], flush=True)
    for m in ("v", "o", "q", "k", "gate", "up", "down"): print("  ", m, res[name]["parts"][m], flush=True)
torch.manual_seed(0); held.clear(); c = env.TIERS[tier]; res["init"] = dict(parts=stats(LM(c, 1000, True, False, 0))); print("init", {m: res["init"]["parts"][m]["stable_rank"] for m in res["init"]["parts"]})
os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=0)
