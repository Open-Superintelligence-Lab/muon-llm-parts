"""Best v,o arms joined with head-muon on q,k, width 48, 1,000 steps, worlds 5-9. No retuning: each part keeps its own tuned lr and momentum."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import map_parts as mp
import arms


def main():
    mp.PATH = os.path.abspath("results/papers/w48x1000.jsonl"); held = [5, 6, 7, 8, 9]; b = arms.parse([json.loads(l) for l in open("results/add/w48x1000_out.jsonl") if "gaps" in l][-1]["base"]); R = mp.Runner("5sx4", 9, .1)
    head = [r for r in json.load(open("results/random/w48x1000_search_out.json"))["held"]["qk"] if r["name"] == "head-muon"][0]; q = arms.parse(head["spec"]); sweep = json.load(open("results/papers/w48x1000_sweep_vo.json"))["rows"]; kaon = json.load(open("results/papers/w48x1000_kaon.json"))["rows"]
    vo = {"exact c=.5": sweep["vo c=0.5"], "c=.7": sweep["vo c=0.7"], "kaon": kaon["vo kaon"]}; R.need([b], held); B = np.array([R.val(b, w) for w in held]); out = {}
    for name, r in vo.items():
        v = arms.parse(r["spec"]); s = {m: (v[m] if m in ("v", "o") else q[m] if m in ("q", "k") else g) for m, g in b.items()}; R.need([s], held); g = B - np.array([R.val(s, w) for w in held]); out[name] = dict(mean=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / 5 ** .5), 4), wins=int((g > 0).sum()), spec=arms.show(s)); print(f"head-muon on q,k + {name} on v,o: {g.mean():+.4f} error {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5 (v,o alone {r['mean']:+.4f}, q,k alone {head['mean']:+.4f})", flush=True)
    json.dump(out, open("results/papers/w48x1000_combo.json", "w"), indent=0); R.ex.shutdown()


if __name__ == "__main__":
    main()
