"""The only way an invented optimizer gets into training. It lives in optimizers/x_name.py and defines step(w, g, st, lr, k).

It sees its weight, that weight's gradient, its own state, its lr and its numeric knobs. It never gets the model, the data, the loss, the step's batch or another weight, so it cannot take a second gradient or read a test row.
Static checks refuse files, imports, dunders, reflection, shared state and memorised numbers before anything runs. While it runs, an audit hook refuses file, network and process calls, and its state is capped at 3x its weight.
"""
import ast, hashlib, os, sys, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); DIR = os.environ.get("OPTIMIZERS_DIR", os.path.join(HERE, "optimizers"))
MAX_BYTES, MAX_STRING, MAX_NUMBERS, MAX_STATE = 6000, 40, 200, 3
IMPORTS = {"torch", "math"}
NAMES = {"open", "eval", "exec", "compile", "getattr", "setattr", "delattr", "hasattr", "globals", "locals", "vars", "dir", "input", "breakpoint", "type", "object", "super", "memoryview", "print", "exit", "quit", "help", "classmethod", "staticmethod", "property", "id"}
ATTRS = {"load", "save", "from_file", "hub", "jit", "package", "onnx", "multiprocessing", "distributed", "utils", "ops", "overrides", "testing", "compiler", "backends", "storage", "untyped_storage", "tofile", "fromfile", "numpy", "backward", "grad", "grad_fn", "manual_seed", "seed", "random", "data", "library", "classes", "modules", "serialization", "autograd", "nn", "fx", "sparse", "profiler", "monitor", "mps", "cpu"}
ATTR_PREFIX = ("_", "set_", "get_", "register_")
BANNED_NODES = (ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.Delete, ast.Try, ast.With, ast.Raise, ast.AsyncFor, ast.AsyncWith)
MODULE_NAMES = {"os", "sys", "gc", "inspect", "subprocess", "importlib", "builtins", "ctypes", "threading", "platform", "functools", "glob", "textwrap", "pickle", "shutil", "socket", "traceback", "warnings", "weakref", "types"}
FRAME_ATTRS = {"f_back", "f_locals", "f_globals", "f_builtins", "f_code", "gi_frame", "gi_code", "cr_frame", "ag_frame", "tb_frame", "tb_next", "format", "format_map", "compile", "vmap", "func"}
TORCH_MODULES = {"linalg", "fft", "special"}  # the only torch submodules an optimizer may reach
BLOCKED = ("sys._getframe", "sys._current_frames", "sys.settrace", "sys.setprofile", "gc.", "object.__getattr__", "object.__setattr__", "object.__delattr__", "open", "os.", "socket.", "subprocess.", "ctypes.", "shutil.", "urllib.", "http.", "ftplib.", "smtplib.", "pty.", "glob.", "tempfile.", "pickle.", "marshal.", "webbrowser.", "sqlite3.", "exec", "compile", "import")
ACTIVE = [False]; HOOKED = [False]; CACHE = {}


class Refused(Exception):
    pass


def hook(event, args):
    if ACTIVE[0] and event.startswith(BLOCKED) and not (event == "import" and str(args[0]).split(".")[0] == "torch"): raise Refused(f"blocked while the optimizer runs: {event}")


def torch_only(name, *a, **k):
    """torch's own C code imports torch the first time a call runs, using the caller's builtins. Nothing else may import, and the file check refuses this name in optimizer code."""
    if name.split(".")[0] != "torch": raise Refused(f"blocked while the optimizer runs: import {name}")
    return __import__(name, *a, **k)


def path_of(name): return os.path.join(DIR, name + ".py")


def code_hash(name): return hashlib.sha256(open(path_of(name), "rb").read()).hexdigest()[:16]


def check(text):
    """Returns the reason this file is refused, or None."""
    if len(text.encode()) > MAX_BYTES: return f"over {MAX_BYTES} bytes"
    try: tree = ast.parse(text)
    except SyntaxError as e: return f"syntax error: {e}"
    numbers = 0
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str): continue  # docstring
        if isinstance(n, ast.Import) and all(a.name in IMPORTS and a.asname is None for a in n.names): continue
        if isinstance(n, ast.FunctionDef): continue
        if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, (int, float)): continue
        return f"line {n.lineno}: only imports of {sorted(IMPORTS)}, plain number constants and functions are allowed at the top"
    tops = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "step"]
    if len(tops) != 1: return "needs exactly one def step(w, g, st, lr, k)"
    if [a.arg for a in tops[0].args.args] != ["w", "g", "st", "lr", "k"]: return "step must take exactly (w, g, st, lr, k)"
    docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    bases, names = set(), []
    for n in ast.walk(tree):
        line = getattr(n, "lineno", 0)
        if isinstance(n, BANNED_NODES): return f"line {line}: {type(n).__name__} is not allowed"
        if isinstance(n, (ast.Import, ast.ImportFrom)) and n not in tree.body: return f"line {line}: import inside a function"
        if isinstance(n, ast.ImportFrom): return f"line {line}: from-imports are not allowed"
        if isinstance(n, ast.Name) and (n.id in NAMES or n.id in MODULE_NAMES or n.id.startswith("_")): return f"line {line}: name {n.id} is not allowed"
        if isinstance(n, ast.Attribute) and (n.attr in ATTRS or n.attr in MODULE_NAMES or n.attr in FRAME_ATTRS or n.attr.startswith(ATTR_PREFIX)): return f"line {line}: attribute {n.attr} is not allowed"
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "torch" and n.attr not in TORCH_MODULES and isinstance(torch.__dict__.get(n.attr), types.ModuleType): return f"line {line}: torch.{n.attr} is a module and is not allowed"
        if isinstance(n, ast.Attribute): bases.add(id(n.value))
        if isinstance(n, ast.Name) and n.id in IMPORTS: names.append(n)
        if isinstance(n, (ast.FunctionDef, ast.Lambda)) and (n.args.defaults or n.args.kw_defaults) and not all(isinstance(d, ast.Constant) for d in n.args.defaults + [x for x in n.args.kw_defaults if x]): return f"line {line}: only plain constants as default arguments"
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                for x in ast.walk(t):
                    if isinstance(x, ast.Attribute): return f"line {line}: assigning to an attribute is not allowed"
        if isinstance(n, ast.Constant) and id(n) not in docs:
            if isinstance(n.value, (str, bytes)) and len(n.value) > MAX_STRING: return f"line {line}: string over {MAX_STRING} characters"
            if isinstance(n.value, (int, float, complex)): numbers += 1
    for n in names:
        if id(n) not in bases: return f"line {n.lineno}: {n.id} may only be used as {n.id}.something"
    if numbers > MAX_NUMBERS: return f"{numbers} numbers in the file, the limit is {MAX_NUMBERS}"
    return None


def state_ok(w, st):
    size = 0
    for v in st.values():
        if torch.is_tensor(v): size += v.numel()
        elif not isinstance(v, (int, float, bool)): raise Refused(f"state holds a {type(v).__name__}; state may hold only tensors and numbers")
    if size > MAX_STATE * w.numel(): raise Refused(f"state holds {size} numbers, over {MAX_STATE}x the weight's {w.numel()}")


def wrap(fn):
    def rule(w, g, st, lr, k):
        knobs = {x: v for x, v in k.items() if x not in ("params", "rule", "lr")}  # a copy without the module's other weights, so nothing is shared
        ACTIVE[0] = True
        try: fn(w.detach(), g.detach().clone(), st, lr, knobs)  # the same numbers with no gradient, hooks or graph attached
        finally: ACTIVE[0] = False
        state_ok(w, st)
    return rule


def load(name, static=True):
    """static=False is only for debug/cheats.py, which tests the runtime layer on its own."""
    if name in CACHE and static: return CACHE[name]
    text = open(path_of(name)).read(); why = check(text) if static else None
    if why: raise Refused(f"{name}: {why}")
    if not HOOKED[0]: sys.addaudithook(hook); HOOKED[0] = True
    keep = ["abs", "min", "max", "sum", "range", "len", "float", "int", "bool", "round", "enumerate", "zip", "isinstance", "tuple", "list", "dict", "__import__"] + ([] if static else ["open"])
    real = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__); space = {"__builtins__": {x: real[x] for x in keep}}
    exec(compile(text, path_of(name), "exec"), space)
    space["__builtins__"]["__import__"] = torch_only  # the real one was for the top of the file only
    f = wrap(space["step"])
    if static: CACHE[name] = f
    return f
