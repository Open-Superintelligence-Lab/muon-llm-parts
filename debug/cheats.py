"""Tries to cheat. Every cheat must be refused, and the example optimizer must still run the same as Muon. Any failure raises. Each stage is stopped at 7 s.

python3 debug/cheats.py
"""
import json, os, signal, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); D = tempfile.mkdtemp(); os.environ["OPTIMIZERS_DIR"] = D; os.environ["SEAL_DIR"] = tempfile.mkdtemp()
import shutil, torch
import arms, env, gate, rules, sandbox

STATIC = {
    "reads a file": "def step(w, g, st, lr, k):\n    open('/tmp/x')\n",
    "imports os": "import os\ndef step(w, g, st, lr, k):\n    pass\n",
    "imports inside": "def step(w, g, st, lr, k):\n    import os\n",
    "from-import": "from os import path\ndef step(w, g, st, lr, k):\n    pass\n",
    "dunder import": "def step(w, g, st, lr, k):\n    __import__('os')\n",
    "class walk": "def step(w, g, st, lr, k):\n    w.__class__.__base__\n",
    "getattr": "def step(w, g, st, lr, k):\n    getattr(w, 'grad')\n",
    "eval": "def step(w, g, st, lr, k):\n    eval('1')\n",
    "torch.load": "import torch\ndef step(w, g, st, lr, k):\n    torch.load('/tmp/x')\n",
    "torch.from_file": "import torch\ndef step(w, g, st, lr, k):\n    torch.from_file('/tmp/x')\n",
    "tofile": "def step(w, g, st, lr, k):\n    w.numpy().tofile('/tmp/x')\n",
    "weight hook": "def step(w, g, st, lr, k):\n    w.register_hook(abs)\n",
    "second gradient": "def step(w, g, st, lr, k):\n    w.backward()\n",
    "global state": "def step(w, g, st, lr, k):\n    global M\n    M = 1\n",
    "shared default": "def step(w, g, st, lr, k, m=[]):\n    m.append(w)\n",
    "function attribute": "def step(w, g, st, lr, k):\n    step.m = w\n",
    "shared list at top": "M = []\ndef step(w, g, st, lr, k):\n    M.append(w)\n",
    "memorised numbers": "def step(w, g, st, lr, k):\n    x = [" + ",".join(["1.5"] * 300) + "]\n",
    "memorised string": "def step(w, g, st, lr, k):\n    x = '" + "a" * 60 + "'\n",
    "huge file": "def step(w, g, st, lr, k):\n    pass\n" + "# pad\n" * 2000,
    "extra arguments": "def step(w, g, st, lr, k, model):\n    pass\n",
    "no step": "def run(w):\n    pass\n",
    "class": "class A:\n    pass\ndef step(w, g, st, lr, k):\n    pass\n",
    "frame walk through torch": "import torch\ndef step(w, g, st, lr, k):\n    torch.inspect.currentframe().f_back.f_locals\n",
    "torch.sys": "import torch\ndef step(w, g, st, lr, k):\n    torch.sys.argv\n",
    "torch.os": "import torch\ndef step(w, g, st, lr, k):\n    torch.os.environ\n",
    "torch.ctypes": "import torch\ndef step(w, g, st, lr, k):\n    torch.ctypes.CDLL\n",
    "torch under another name": "import torch\ndef step(w, g, st, lr, k):\n    t = torch\n    t.linalg\n",
    "torch passed as a value": "import torch\ndef f(x):\n    return x\ndef step(w, g, st, lr, k):\n    f(torch)\n",
    "format reaches attributes": "def step(w, g, st, lr, k):\n    '{0.grad}'.format(w)\n",
    "generator frame": "def step(w, g, st, lr, k):\n    (x for x in [1]).gi_frame\n",
    "traceback frame": "def step(w, g, st, lr, k):\n    w.tb_frame\n",
}
RUNTIME = {
    "open while running": "def step(w, g, st, lr, k):\n    open('/etc/hosts')\n",
    "import while running": "def step(w, g, st, lr, k):\n    import socket\n",
    "shell command": "import os\ndef step(w, g, st, lr, k):\n    os.system('true')\n",
    "lists a folder": "import os\ndef step(w, g, st, lr, k):\n    os.listdir('.')\n",
    "opens a socket": "import socket\ndef step(w, g, st, lr, k):\n    socket.socket()\n",
    "reads the caller's frame": "import sys\ndef step(w, g, st, lr, k):\n    sys._getframe(1)\n",
    "lists all live objects": "import gc\ndef step(w, g, st, lr, k):\n    gc.get_objects()\n",
    "traces other calls": "import sys\ndef step(w, g, st, lr, k):\n    sys.settrace(None)\n",
}
LIMITS = {
    "state too big": "import torch\ndef step(w, g, st, lr, k):\n    st['a'] = torch.zeros(w.numel() * 4)\n",
    "state hides tensors in a list": "import torch\ndef step(w, g, st, lr, k):\n    st['a'] = [w.clone()]\n",
}


def write(name, text):
    with open(os.path.join(D, name + ".py"), "w") as f: f.write(text)


class stage:
    def __init__(s, name): s.name = name
    def __enter__(s): signal.signal(signal.SIGALRM, lambda *_: sys.exit(f"{s.name} ran over 7 s")); signal.alarm(7)
    def __exit__(s, kind, *rest):
        signal.alarm(0)
        if not kind: print("ok", s.name, flush=True)


def blocked(f):
    try: f()
    except (sandbox.Refused, ImportError): return True
    return False


W = lambda: torch.randn(8, 4)
with stage(f"{len(STATIC)} files refused before running"):
    for what, text in STATIC.items():
        assert sandbox.check(text), f"not refused: {what}"
with stage("the audit hook stops file and import calls even if the file check is skipped"):
    for i, (what, text) in enumerate(RUNTIME.items()):
        write(f"x_rt{i}", text); f = sandbox.load(f"x_rt{i}", static=False); w = W()
        assert blocked(lambda: f(w, W(), {}, .1, {})), f"not blocked: {what}"
with stage("state limits hold while running"):
    for i, (what, text) in enumerate(LIMITS.items()):
        write(f"x_lim{i}", text); f = sandbox.load(f"x_lim{i}"); w = W()
        assert blocked(lambda: f(w, W(), {}, .1, {})), f"not blocked: {what}"
with stage("an optimizer gets no other weights and cannot change the gradient"):
    write("x_look", "def step(w, g, st, lr, k):\n    st['n'] = len(k)\n    st['p'] = 'params' in k\n    g.mul_(0)\n")
    w, g, st = W(), W(), {}; keep = g.clone(); sandbox.load("x_look")(w, g, st, .1, dict(params=[w], rule="x_look", lr=.1, t=False)); assert not st["p"] and st["n"] == 1 and (g == keep).all()
with stage("an optimizer gets a bare weight with no gradient or graph"):
    write("x_bare", "def step(w, g, st, lr, k):\n    st['leaf'] = w.requires_grad\n    w.add_(g, alpha=-lr)\n")
    w = torch.nn.Parameter(torch.ones(4, 2)); w.grad = torch.ones(4, 2); st = {}; keep = w.detach().clone(); sandbox.load("x_bare")(w, torch.ones(4, 2), st, .5, {}); assert st["leaf"] is False and (w.detach() == keep - .5).all()
with stage("the example optimizer takes Muon's exact steps"):
    shutil.copy(os.path.join(HERE, "optimizers", "x_muon_copy.py"), D); f = sandbox.load("x_muon_copy"); torch.manual_seed(0)
    for m, shape, t in (("qk", (8, 4), False), ("down", (8, 16), True)):
        a = torch.randn(*shape); b = a.clone(); s1, s2 = {}, {}
        for _ in range(3):
            g = torch.randn(*shape); rules.muon(a, g, s1, .02, dict(t=t)); f(b, g, s2, .02, dict(t=t))
        assert torch.allclose(a, b, atol=1e-5), f"{m} differs by {(a - b).abs().max()}"
with stage("the gate refuses each shortcut"):
    write("x_ok", "def step(w, g, st, lr, k):\n    w.add_(g, alpha=-lr)\n")
    s = dict(worlds=[1, 2, 3, 4, 8], code=gate.hashes()); base = dict(joint=True, kind="tune", arm="muon", spec="emb=adam:.3;head=adam:.03;qk=muon:.02;vo=muon:.02;gu=muon:.01;down=muon:.02;norm=adam:.02", tune=4.4, evals=60, tier="5s", moe=0)
    spec = "emb=adam:.3;head=adam:.03;qk=x_ok:.02;vo=muon:.02;gu=muon:.01;down=muon:.02;norm=adam:.02"; design = dict(joint=True, kind="custom", arm="x_ok", spec=spec, tune=4.3, evals=60, tier="5s", moe=0)
    a = env.parser().parse_args([]); a.__dict__.update(tier="5s", world=0, evals="tune"); key = env.key_of(a, arms.parse(spec))
    assert gate.refuse("x_ok", "5s", s, [base, design], {key})[0] is None, "an honest design was refused"
    assert "locked code" in gate.refuse("x_ok", "5s", s, [base, design], {key}, now=dict(gate.hashes(), **{"evals.py": "edited"}))[0], "edited evaluation passed"
    assert "no x_other" in gate.refuse("x_other", "5s", s, [base, design], {key})[0], "a design that never went through tuning passed"
    assert "no tune row" in gate.refuse("x_ok", "5s", s, [base, design], set())[0], "a typed choices row passed"
    write("x_ok", "def step(w, g, st, lr, k):\n    w.add_(g, alpha=-2 * lr)\n")
    assert "no tune row" in gate.refuse("x_ok", "5s", s, [base, design], {key})[0], "an optimizer swapped after tuning passed"
    write("x_ok", "def step(w, g, st, lr, k):\n    w.add_(g, alpha=-lr)\n")
    assert "no x_ok" in gate.refuse("x_ok", "5s", s, [base, dict(design, joint=False)], {key})[0], "a design without momentum tuning passed"
    assert "no opponent" in gate.refuse("x_ok", "5s", s, [dict(base, joint=False), design], {key})[0], "an opponent without momentum tuning was used"
    more = dict(design, evals=300); k = gate.refuse("x_ok", "5s", s, [base, more], {key})
    assert k[0] and "settings" in k[0], "a design tuned 5x longer than its opponent passed"
    write("x_ok", "import os\ndef step(w, g, st, lr, k):\n    pass\n"); assert "refused" in gate.refuse("x_ok", "5s", s, [base, design], {key})[0], "a bad file passed at the gate"
with stage("the threshold rises with every design submitted"):
    from scipy import stats
    assert stats.t.ppf(1 - gate.ALPHA / 2 / 20, 4) > 1.5 * stats.t.ppf(1 - gate.ALPHA / 2, 4)
with stage("search can reach only world 0 and tune rows"):
    import search
    x = search.spec_args(arms.parse("muon"), env.parser().parse_args(["--tier", "5s"])); assert x.world == 0 and x.evals == "tune" and not x.final
print("all cheats refused")
