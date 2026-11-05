"""Packs one private Kaggle job of debug/papers_parts.py: python3 kaggle/make_papers.py NAME TIER BASE_OUT SEED "args for one run" ["args for the next run" ...]. Push with kaggle kernels push -p kaggle/NAME."""
import base64, io, json, os, sys, tarfile
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE); name, tier, base, seed, *runs = sys.argv[1:]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ("env.py", "evals.py", "rules.py", "arms.py", "runq.py", "sandbox.py", "debug/map_parts.py", "debug/papers_parts.py", "optimizers/x_rand.py", "optimizers/x_kaon.py", "optimizers/x_dion3.py", "optimizers/x_nsteps.py", "optimizers/x_band.py", "optimizers/x_hyc.py"): t.add(os.path.join(SRC, f), f)
    for f in ("facts.json", "counts.npy"): t.add(os.path.realpath(os.path.join(SRC, "data", f)), "data/" + f)
    t.add(base, "base_out.jsonl")
job = os.path.join(HERE, name); os.makedirs(job, exist_ok=True)
body = f"CODE = {base64.b64encode(buf.getvalue()).decode()!r}\nSEED = {base64.b64encode(open(seed, 'rb').read()).decode()!r}\nTIER = {tier!r}\nRUNS = {[r.split() for r in runs]!r}\nHOURS = 11.3\n" + r'''
import base64, io, os, subprocess, sys, tarfile, time
T0 = time.time(); OUT = "/kaggle/working"; C = "/tmp/code"; W = "/tmp/web"; os.makedirs(C, exist_ok=True); os.makedirs(W, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall(C)
open(OUT + "/map.jsonl", "wb").write(base64.b64decode(SEED))
log = open(OUT + "/log.txt", "a", buffering=1)
def say(*a):
    print(*a, flush=True); print(*a, file=log)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=True)
WEB = "https://huggingface.co/datasets/kjj0/fineweb100B-gpt2/resolve/main/"
for f in ["fineweb_val_000000.bin"] + [f"fineweb_train_{w + 1:06d}.bin" for w in (0, 1, 5, 6, 7, 8, 9)]:
    if not os.path.exists(f"{W}/{f}"): subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", f"{W}/{f}", WEB + f], check=True)
say("data ready", round(time.time() - T0), "s"); say(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
E = dict(os.environ, FWDIR=W, RUNS_DIR="/tmp/runs", PYTHONWARNINGS="ignore", LOAD="99"); E.pop("OPTIMIZERS_DIR", None)
for i, r in enumerate(RUNS):
    left = HOURS - (time.time() - T0) / 3600
    if left < .1: say("out of time before", r); break
    say("==", r, "hours left", round(left, 2))
    p = subprocess.Popen([sys.executable, "debug/papers_parts.py", TIER, "base_out.jsonl", "--results", OUT + "/map.jsonl", "--out", OUT + f"/out{i}.json", "--hours", str(left), "--par", "4"] + r, cwd=C, env=E, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if not any(x in line for x in ("Exception ignored", "weakref", "AttributeError")): say(line.rstrip())
    p.wait(); say(r, "done with code", p.returncode, "at", round((time.time() - T0) / 60), "min")
'''
open(os.path.join(job, "job.py"), "w").write(body)
json.dump({"id": f"{json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username']}/muon-parts-{name}", "title": f"muon-parts-{name}", "code_file": "job.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open(os.path.join(job, "kernel-metadata.json"), "w"), indent=1)
print("wrote", job, round(len(body) / 1e6, 2), "MB")
