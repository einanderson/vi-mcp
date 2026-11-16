"""Structure facts of code (0.66.0): what a Python file defines, calls, imports, writes, raises — read by clingo-object rules.

The user (15.11.2026): code text stays text (code_import's pieces); from the same `ast` pass the importer also derives the structure,
so rules can answer "which functions write table objekt", "what calls memory.save", "is there an import cycle" exactly, where a text
search can only guess.

Design choice — the facts are hashtags of memory objects, read by the existing path (logic.object_data → notation.facts →
vi__tag), not a second fact store. Reason: one data model (objects with hashtags, owning instance, visibility, index, archive with
tombstones) and one import path; a fact table of its own would need its own visibility, incremental sync and deletion rules, and
clingo-object would need a second syntax to read it. One fact object per function/method (`code:funktion`), per module
(`code:modul`) and per literal module constant (`code:konstante`), next to the text pieces and with the same `knowledge_kind:code`, so
everything that keeps code out of hints, training data and network exports keeps the facts out too. Per-object facts are only
LOCAL facts of one file (what this function calls); relations across files (tested_by, import cycles) are derived by rules — so an
incremental sync of one changed file never has to touch another file's objects.

Facts per object (hashtag kinds, German as all stored formats):
  code:funktion  function:<module.qualname> module: file: calls:<callee> calls_locked:<callee> writes:<table>
                 writes_unlocked:<table> raises:<exception> test:ja body:<hash> changed_in:<commit12>
  code:modul     module: file: imports:<module> (at import time) imports_late:<module> (inside a function)
  code:konstante constant:<module.NAME> module: file: key:<key> value:<value>

Stated limits (each is a class of miss, counted where it can be): calls are resolved through imports, module-level definitions and
self/cls; a call on a local object (o.change()) is not resolved and only counted (`unaufgeloest`); SQL is found in string constants
whose text names the table (an f-string table name is not found); "gesperrt" is lexical — inside `with <x>.lock` in the same
function; `koerper` is the hash of the exact body (renamed variables make another hash); `geaendert_in` is from `git blame` of the
working tree: the commits that last touched a line of the function (at most HISTORY, newest first).
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import re
import subprocess
from pathlib import Path

#: SQL that changes a table; `DO UPDATE SET` (upsert) is part of an INSERT and names no table
WRITE_SQL = re.compile(r"\b(?:INSERT\s+(?:OR\s+[A-Z]+\s+)?INTO|REPLACE\s+INTO|(?<!DO )UPDATE|DELETE\s+FROM)\s+([A-Za-z_][A-Za-z0-9_]*)")
LOCK = re.compile(r"(^|[._])lock$")
VALUE = re.compile(r"^[\w.:/@-]{1,120}$")
BUILTINS = frozenset(dir(builtins))
MIN_BODY_NODES = 12  # a body smaller than this (a one-line wrapper) is no duplicate worth naming
MAX_PER_CONSTANT = 400
#: the hashtag that marks a fact object (next to art_wissen:code), one per kind of object
FACT_TAGS = ("code:funktion", "code:modul", "code:konstante")
HISTORY = 5


def module_name(path: str) -> str:
    p = path[:-3] if path.endswith(".py") else path
    parts = [x for x in p.replace("\\", "/").split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def python_modules(paths) -> set[str]:
    return {module_name(p) for p in paths if p.endswith(".py")}


def is_test(path: str, name: str) -> bool:
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    return (path.startswith("tests/") or "/tests/" in path or base.startswith("test_")) and name.rsplit(".", 1)[-1].startswith("test")


class _Names:
    """What a name means in one module: imported modules and names, module-level definitions."""

    def __init__(self, module: str, is_package: bool, modules: set[str]):
        self.module, self.modules = module, modules
        self.package = module if is_package else module.rpartition(".")[0]
        self.alias: dict[str, str] = {}
        self.defs: set[str] = set()

    def base(self, node: ast.ImportFrom) -> str:
        if not node.level:
            m = node.module or ""
            if m and m not in self.modules:  # tests import siblings by bare name (from test_p2p import …): tests.test_p2p
                same = [x for x in self.modules if x.endswith("." + m) and x.rsplit(".", 1)[0] == self.package]
                m = same[0] if len(same) == 1 else m
            return m
        pkg = self.package.split(".") if self.package else []
        pkg = pkg[:len(pkg) - (node.level - 1)] if node.level > 1 else pkg
        return ".".join(pkg + ([node.module] if node.module else []))

    def bind(self, node, table: dict[str, str]) -> list[str]:
        """Record the names an import binds into table; return the modules it imports."""
        out = []
        if isinstance(node, ast.Import):
            for a in node.names:
                table[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                out.append(a.name)
        else:
            base = self.base(node)
            for a in node.names:
                if a.name == "*":
                    out.append(base)
                    continue
                full = f"{base}.{a.name}" if base else a.name
                table[a.asname or a.name] = full
                out.append(full if full in self.modules else base)
        return [m for m in out if m]

    def resolve(self, expr, local: dict[str, str], cls: str | None) -> str | None:
        parts = []
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name):
            return None
        root = expr.id
        if root in ("self", "cls") and cls and parts:
            head = cls
        elif root in local:
            head = local[root]
        elif root in self.alias:
            head = self.alias[root]
        elif root in self.defs:
            head = f"{self.module}.{root}"
        elif root in BUILTINS and not parts:
            return root
        else:
            return None
        return ".".join([head] + parts[::-1])


def _parents(root) -> dict:
    out = {}
    for n in ast.walk(root):
        for c in ast.iter_child_nodes(n):
            out[c] = n
    return out


def _locked(node, parents: dict, stop) -> bool:
    n = parents.get(node)
    while n is not None and n is not stop:
        if isinstance(n, (ast.With, ast.AsyncWith)) and any(LOCK.search(ast.unparse(i.context_expr)) for i in n.items):
            return True
        n = parents.get(n)
    if n is stop and isinstance(stop, (ast.With, ast.AsyncWith)):
        return any(LOCK.search(ast.unparse(i.context_expr)) for i in stop.items)
    return False


def _docstring_node(fn):
    b = getattr(fn, "body", [])
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
        return b[0].value
    return None


def _body_hash(fn) -> str | None:
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
    if sum(1 for s in body for _ in ast.walk(s)) < MIN_BODY_NODES:
        return None
    dump = "\n".join(ast.dump(s, annotate_fields=False, include_attributes=False) for s in body)
    return hashlib.sha256(dump.encode()).hexdigest()[:16]


def _function(fn, qual: str, cls: str | None, names: _Names, path: str) -> dict:
    parents = _parents(fn)
    local: dict[str, str] = {}
    late: list[str] = []
    for n in ast.walk(fn):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            late += names.bind(n, local)
    calls, calls_locked, writes, unlocked, raises = set(), set(), set(), set(), set()
    unresolved = 0
    doc = _docstring_node(fn)
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            target = names.resolve(n.func, local, cls)
            if target is None:
                unresolved += 1
            elif target not in BUILTINS:
                calls.add(target)
                if _locked(n, parents, fn):
                    calls_locked.add(target)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n is not doc:
            for table in WRITE_SQL.findall(n.value):
                writes.add(table)
                if not _locked(n, parents, fn):
                    unlocked.add(table)
        elif isinstance(n, ast.Raise) and n.exc is not None:
            exc = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
            target = names.resolve(exc, local, cls)
            if target:
                raises.add(target)
    return {"name": qual, "symbol": qual[len(names.module) + 1:] if qual.startswith(names.module + ".") else qual,
            "von": min([fn.lineno] + [d.lineno for d in fn.decorator_list]), "bis": fn.end_lineno or fn.lineno,
            "ruft": sorted(calls), "ruft_gesperrt": sorted(calls_locked), "schreibt": sorted(writes), "schreibt_ungesperrt": sorted(unlocked),
            "wirft": sorted(raises), "test": is_test(path, qual), "koerper": _body_hash(fn), "unaufgeloest": unresolved,
            "importiert_spaet": sorted(set(late))}


def _constant(node, module: str) -> dict | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name, value = node.targets[0].id, node.value
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        name, value = node.target.id, node.value
    else:
        return None
    keys, vals = [], []

    def text(x):
        return x.value if isinstance(x, ast.Constant) and isinstance(x.value, str) and VALUE.match(x.value) else None

    if isinstance(value, ast.Dict):
        keys = [k for k in (text(k) for k in value.keys if k is not None) if k]
        vals = [v for v in (text(v) for v in value.values) if v]
    elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for e in value.elts:
            if isinstance(e, ast.Dict):  # a list of records: their "name" values (e.g. the tool list)
                vals += [text(v) for k, v in zip(e.keys, e.values) if isinstance(k, ast.Constant) and k.value == "name" and text(v)]
            elif text(e):
                vals.append(text(e))
    if not keys and not vals:
        return None
    return {"name": f"{module}.{name}", "von": node.lineno, "bis": node.end_lineno or node.lineno,
            "schluessel": sorted(set(keys))[:MAX_PER_CONSTANT], "wert": sorted(set(vals))[:MAX_PER_CONSTANT]}


def structure(path: str, src: str, modules: set[str]) -> dict | None:
    """The structure facts of one Python file, or None if it does not parse. `modules`: every Python module of the repository (to tell
    `from pkg import name` a module from a name)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    module = module_name(path)
    names = _Names(module, path.endswith("__init__.py"), modules | {module})
    top_imports: list[str] = []
    functions: list[dict] = []
    constants: list[dict] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.defs.add(n.name)
    fn_nodes = set()
    for n in ast.walk(tree):  # import time: every import outside a function (module level, if/try, class bodies), not under TYPE_CHECKING
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            fn_nodes.update(ast.walk(n))
    parents = _parents(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)) and n not in fn_nodes:
            p, typing_only = parents.get(n), False
            while p is not None:
                if isinstance(p, ast.If) and "TYPE_CHECKING" in ast.unparse(p.test):
                    typing_only = True
                p = parents.get(p)
            mods = names.bind(n, names.alias)
            if not typing_only:
                top_imports += mods
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_function(n, f"{module}.{n.name}", None, names, path))
        elif isinstance(n, ast.ClassDef):
            cls = f"{module}.{n.name}"
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(_function(m, f"{cls}.{m.name}", cls, names, path))
        else:
            c = _constant(n, module)
            if c:
                constants.append(c)
    late = sorted({m for f in functions for m in f.pop("importiert_spaet")} - set(top_imports))
    return {"modul": module, "importiert": sorted(set(top_imports)), "importiert_spaet": late, "funktionen": functions,
            "konstanten": constants}


def blame(repo: Path, path: str) -> dict[int, tuple[str, int]]:
    """{line: (commit, committer time)} from `git blame --porcelain` of the working tree; uncommitted lines are left out."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "blame", "--porcelain", "--", path], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", check=True, timeout=120).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return {}
    lines, times, commit = {}, {}, None
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line)
        if m:
            commit = m.group(1)
            if not commit.strip("0"):
                commit = None
            else:
                lines[int(m.group(2))] = commit
        elif line.startswith("committer-time ") and commit:
            times[commit] = int(line.split()[1])
    return {ln: (c, times.get(c, 0)) for ln, c in lines.items()}


def add_history(st: dict, lines: dict[int, tuple[str, int]]) -> None:
    for f in st["funktionen"]:
        seen = {lines[ln] for ln in range(f["von"], f["bis"] + 1) if ln in lines}
        f["geaendert_in"] = [c[:12] for c, _t in sorted(seen, key=lambda x: (-x[1], x[0]))[:HISTORY]]  # same second: by hash


# ── the objects (server side) ─────────────────────────────────────────────────────────────────────────────────────────────────────

def objects(st: dict, path: str) -> list[tuple[str, str, list[str], dict]]:
    """(symbol for the id, text, fact hashtags, data) of every fact object of one file's structure."""
    from .code_import import tag_value as tv
    mod = st["modul"]
    out = []
    head = [f"module:{tv(mod)}"]
    out.append(("#modul", f"{path} — module {mod}: imports {len(st['importiert'])} at import time",
                ["code:modul"] + head + [f"imports:{tv(m)}" for m in st["importiert"]]
                + [f"imports_late:{tv(m)}" for m in st["importiert_spaet"]],
                {"modul": mod, "importiert": st["importiert"], "importiert_spaet": st["importiert_spaet"]}))
    for f in st["funktionen"]:
        tags = ["code:funktion", f"function:{tv(f['name'])}"] + head
        tags += [f"calls:{tv(x)}" for x in f["ruft"]] + [f"calls_locked:{tv(x)}" for x in f["ruft_gesperrt"]]
        tags += [f"writes:{tv(x)}" for x in f["schreibt"]] + [f"writes_unlocked:{tv(x)}" for x in f["schreibt_ungesperrt"]]
        tags += [f"raises:{tv(x)}" for x in f["wirft"]] + (["test:ja"] if f["test"] else [])
        tags += [f"body:{f['koerper']}"] if f.get("koerper") else []
        tags += [f"changed_in:{c}" for c in f.get("geaendert_in") or []]
        text = (f"{path}:{f['symbol']} — calls {len(f['ruft'])}" + (f", writes {', '.join(f['schreibt'])}" if f["schreibt"] else "")
                + (f", raises {', '.join(x.rsplit('.', 1)[-1] for x in f['wirft'])}" if f["wirft"] else ""))
        out.append((f"#function:{f['symbol']}", text[:300], tags, {k: v for k, v in f.items() if k != "symbol"}))  # data.symbol = a piece
    for c in st["konstanten"]:
        tags = ["code:konstante", f"constant:{tv(c['name'])}"] + head
        tags += [f"key:{tv(x)}" for x in c["schluessel"]] + [f"value:{tv(x)}" for x in c["wert"]]
        out.append((f"#constant:{c['name']}", f"{path} — constant {c['name']}: {len(c['schluessel'])} keys, {len(c['wert'])} values",
                    tags, dict(c)))
    return out


# ── the bridge: fact objects → the input predicates of the `code` philosophy ──────────────────────────────────────────────────────

def bridge(repo: str) -> str:
    """clingo-object rules that read the fact objects of one repository (any index the asking instance may see) as the input
    predicates of the `code` philosophy package. The seed itself never reads objects — its worlds are plain facts."""
    r = repo.replace('"', "")
    sel = f'knowledge_kind:code repo:{r}'
    return f"""\
snapshot(R) <= (O in objects("{sel} code:modul")) & (O.repo == R)
module(M) <= (O in objects("{sel} code:modul")) & (O.modul == M)
imports(A, B) <= (O in objects("{sel} code:modul")) & (O.modul == A) & (O.importiert == B)
imports_late(A, B) <= (O in objects("{sel} code:modul")) & (O.modul == A) & (O.importiert_spaet == B)
defines(D, F) <= (O in objects("{sel} code:funktion")) & (O.datei == D) & (O.funktion == F)
in_module(F, M) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.modul == M)
calls(F, G) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.ruft == G)
calls_locked(F, G) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.ruft_gesperrt == G)
writes_table(F, T) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.schreibt == T)
writes_unlocked(F, T) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.schreibt_ungesperrt == T)
raises(F, E) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.wirft == E)
test(F) <= (O in objects("{sel} code:funktion test:ja")) & (O.funktion == F)
body(F, H) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.koerper == H)
changed_in(F, C) <= (O in objects("{sel} code:funktion")) & (O.funktion == F) & (O.geaendert_in == C)
const_key(K, V) <= (O in objects("{sel} code:konstante")) & (O.konstante == K) & (O.schluessel == V)
const_value(K, V) <= (O in objects("{sel} code:konstante")) & (O.konstante == K) & (O.wert == V)
"""


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def direct_facts(structures: dict[str, dict], repo: str | None = None) -> list[str]:
    """The same input predicates straight from structures {path: structure} — for worlds, tests and measurements without a memory.
    tests/test_code_facts.py checks that the bridge over stored objects gives exactly these atoms (one meaning, two routes)."""
    out = [f"snapshot({_q(repo)})"] if repo else []  # the structures are every Python file of one commit: a closed view
    for path, st in structures.items():
        m = st["modul"]
        out.append(f"module({_q(m)})")
        out += [f"imports({_q(m)}, {_q(x)})" for x in st["importiert"]]
        out += [f"imports_late({_q(m)}, {_q(x)})" for x in st["importiert_spaet"]]
        for f in st["funktionen"]:
            q = _q(f["name"])
            out += [f"defines({_q(path)}, {q})", f"in_module({q}, {_q(m)})"]
            out += [f"calls({q}, {_q(x)})" for x in f["ruft"]] + [f"calls_locked({q}, {_q(x)})" for x in f["ruft_gesperrt"]]
            out += [f"writes_table({q}, {_q(x)})" for x in f["schreibt"]] + [f"writes_unlocked({q}, {_q(x)})" for x in f["schreibt_ungesperrt"]]
            out += [f"raises({q}, {_q(x)})" for x in f["wirft"]] + ([f"test({q})"] if f["test"] else [])
            out += [f"body({q}, {_q(f['koerper'])})"] if f.get("koerper") else []
            out += [f"changed_in({q}, {_q(c)})" for c in f.get("geaendert_in") or []]
        for c in st["konstanten"]:
            out += [f"const_key({_q(c['name'])}, {_q(x)})" for x in c["schluessel"]] + [f"const_value({_q(c['name'])}, {_q(x)})" for x in c["wert"]]
    return out


def scan(repo: Path, history: bool = False) -> dict[str, dict]:
    """Every versioned Python file of a repository → its structure (for measurements and `vi-mcp code facts`)."""
    from . import code_import
    _commit, files, _skipped = code_import.collect(repo)
    py = sorted(p for p in files if p.endswith(".py"))
    mods = python_modules(py)
    out = {}
    for p in py:
        src, _why = code_import.read_text(repo, p)
        st = structure(p, src, mods) if src is not None else None
        if st is not None:
            if history:
                add_history(st, blame(repo, p))
            out[p] = st
    return out


# ── checking a repository locally ─────────────────────────────────────────────────────────────────────────────────────────────────

def evaluate(structures: dict[str, dict], repo: str, profile: str = "", extra: str = "", timeout: float = 120) -> dict:
    """The `code` philosophy over a scan: seed + profile (+ extra rules, e.g. a question) in clingo-object notation, the facts as ASP
    (they are far above the notation's source limit; they are data, not rules)."""
    from clingo_object import notation as n
    from clingo_object import philosophy_code as pc
    facts = "\n".join(f + "." for f in direct_facts(structures, repo))
    program, stmts, _filters = n.translate_full(pc.SEED + "\n" + profile + "\n" + extra)
    raw = n.run_program(facts + "\n" + program, stmts, n.prior_belief, timeout)
    res = n.check_result(raw, n.prior_belief)  # its folgerungen are cut at a display limit — the atoms below are complete
    models = [set(m) for m in raw.get("models") or []]
    res["atoms"] = sorted(a for a in (set.intersection(*models) if models else set()) if n._visible(a))
    return res


def report(r: dict) -> dict:
    """status per rule, findings with reasons, excused findings with their reasons, duties — from a check result."""
    from clingo_object.philosophy import _args
    got = r.get("atoms", r.get("folgerungen", []))

    def args(pred):
        return [[a.strip('"') for a in _args(x)] for x in got if x.startswith(pred + "(")]

    return {"state": r.get("zustand"), "status": {a[0]: a[1] for a in args("status")},
            "findings": sorted((a[0], a[1], a[2]) for a in args("grund") if [a[0], a[1]] in args("finding")),
            "excused": sorted(tuple(a) for a in args("excused")),
            "add_test": sorted(a[0] for a in args("untested")),
            "violations": [v.get("never") for v in r.get("verletzt", [])]}


def main(argv: list[str]) -> int:
    """vi-mcp index code-check REPO [--profile vi-mcp|FILE|none] [--facts] [--json] — the `code` philosophy over a local repository."""
    import argparse
    import json
    a = argparse.ArgumentParser(prog="vi-mcp index code-check", description="architecture rules (philosophy `code`) over a repository's structure")
    a.add_argument("repo", type=Path)
    a.add_argument("--profile", default="none", help="vi-mcp (this project's profile), a file with profile facts, or none")
    a.add_argument("--facts", action="store_true", help="print the structure facts (ASP) and stop")
    a.add_argument("--history", action="store_true", help="git blame for changed_in (one git call per file)")
    a.add_argument("--json", action="store_true")
    o = a.parse_args(argv)
    st = scan(o.repo, history=o.history)
    name = o.repo.resolve().name
    if o.facts:
        print("\n".join(f + "." for f in direct_facts(st, name)))
        return 0
    from clingo_object import philosophy_code as pc
    profile = pc.PROFILE_VI_MCP if o.profile == "vi-mcp" else "" if o.profile == "none" else Path(o.profile).read_text(encoding="utf-8")
    rep = report(evaluate(st, name, profile))
    if o.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(f"{name}: {len(st)} Python files, {sum(len(s['funktionen']) for s in st.values())} functions — state {rep['state']}")
        for q, v in sorted(rep["status"].items()):
            print(f"  {q}: {v}")
        for q, x, w in rep["findings"]:
            print(f"    finding {q}: {x} — {w}")
        for q, x, w in rep["excused"]:
            print(f"    excused {q}: {x} — {w}")
        if rep["add_test"]:
            print(f"  without a direct test, writing a guarded table: {', '.join(rep['add_test'])}")
    return 0 if rep["state"] == "erfuellt" and all(v == "ok" for v in rep["status"].values()) else 1


# ── training form (design, not exported yet) ──────────────────────────────────────────────────────────────────────────────────────

def training_document(f: dict, path: str, code: str = "") -> str:
    """The grounding document of one function's pairs: its facts as sentences, then its code (the piece) — qa_pairs.check grounds every
    answer in it."""
    lines = [f"{f['name']} in {path}, lines {f['von']}-{f['bis']}"]
    if f.get("ruft"):
        lines.append(f"{f['name']} calls " + ", ".join(f["ruft"][:20]) + ".")
    if f.get("schreibt"):
        lines.append(f"{f['name']} writes the tables " + ", ".join(f["schreibt"]) + ".")
    if f.get("schreibt_ungesperrt"):
        lines.append(f"{f['name']} writes " + ", ".join(f["schreibt_ungesperrt"]) + " outside a lock.")
    elif f.get("schreibt"):
        lines.append(f"{f['name']} writes the tables " + ", ".join(f["schreibt"]) + " only inside the lock.")
    if f.get("wirft"):
        lines.append(f"{f['name']} raises " + ", ".join(f["wirft"]) + ".")
    return "\n".join(lines) + ("\n\n" + code if code else "")


def training_pairs(f: dict, path: str, object_id: str, code: str = "") -> dict:
    """Grounded question-answer pairs about one function from its facts — no language model: the facts are exact, so are the answers.
    Kept, rejected and the probe exactly as for knowledge (vi_mcp.qa_pairs: one definition of a good pair)."""
    from . import qa_pairs as qa
    name, short = f["name"], f["name"].rsplit(".", 1)[-1]
    doc = training_document(f, path, code)
    cand = [{"question": f"Where is {name} defined?", "answer": f"In {path}, lines {f['von']}-{f['bis']}."}]
    if f.get("ruft"):
        callees = ", ".join(f["ruft"][:20])
        cand += [{"question": f"Which functions does {name} call?", "answer": f"{name} calls {callees}."},
                 {"question": f"What does {short} in {path} call directly?", "answer": f"It calls {callees}."}]
    if f.get("schreibt"):
        tables = ", ".join(f["schreibt"])
        cand += [{"question": f"Which database tables does {name} write?", "answer": f"{name} writes the tables {tables}."},
                 {"question": f"Does {short} in {path} change rows of a table?", "answer": f"It writes the tables {tables}."}]
        cand.append({"question": f"Does {name} hold the lock when it writes?",
                     "answer": (f"{name} writes {', '.join(f['schreibt_ungesperrt'])} outside a lock." if f.get("schreibt_ungesperrt")
                                else f"{name} writes the tables {tables} only inside the lock.")})  # no yes/no: those words are in no document
    if f.get("wirft"):
        exc = ", ".join(f["wirft"])
        cand += [{"question": f"Which exceptions can {name} raise?", "answer": f"{name} raises {exc}."},
                 {"question": f"What does {short} in {path} raise on failure?", "answer": f"It raises {exc}."}]
    kept, rejected = qa.select(cand, doc)
    train, probe, dropped = qa.split_probe(object_id, kept)
    return {"schema": qa.SCHEMA_VERSION, "generator": "code-facts-v1", "pairs": train, "probe": probe, "rejected": rejected,
            "dropped_near_probe": dropped, "content_sha256": qa.content_sha256(doc)}
