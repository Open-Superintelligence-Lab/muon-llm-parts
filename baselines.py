"""Train every baseline once with its tuned settings and print the loss on held-out web text, lowest first.

python3 baselines.py [--world 7] [--only muon,all-adam]

Settings are in baselines.jsonl, tuned on world 0 at the 5s tier (width 48, 2 layers, 250 steps) in llm-module-optimizers. They hold for this tier only.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); SCORE = "text.web.loss"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--world", type=int, default=7); p.add_argument("--only", default=""); a = p.parse_args()
    arms = [json.loads(l) for l in open(os.path.join(HERE, "baselines.jsonl"))]; out = os.path.join(HERE, "results.jsonl"); done = {}
    for r in arms:
        if a.only and r["arm"] not in a.only.split(","): continue
        t = time.time(); subprocess.run([sys.executable, "env.py", "--spec", r["spec"], "--tier", r["tier"], "--world", str(a.world), "--evals", "full", "--tag", "baseline"], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
        row = [json.loads(l) for l in open(out)][-1]; done[r["arm"]] = row["scores"].get(SCORE); print(f"{r['arm']:16} {done[r['arm']]}  {time.time() - t:.0f}s", flush=True)
    base = done.get("muon"); print(f"\nworld {a.world}, lower is better, gap is against muon")
    for k, v in sorted(done.items(), key=lambda x: x[1] if x[1] is not None else 1e9):
        print(f"{k:16} {v:.4f}" + (f"  {v - base:+.4f}" if base is not None else ""))


if __name__ == "__main__":
    main()
