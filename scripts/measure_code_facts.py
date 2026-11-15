"""Measure structure questions about this repository: clingo-object over the stored structure facts vs the plain search over the code
pieces, against a ground truth from grep, git and a separate ast pass (never from vi_mcp.code_facts).

    python scripts/measure_code_facts.py SUBJECT_REPO OUT.json

SUBJECT_REPO: a clean checkout of this repository (no uncommitted changes: git blame and git diff must agree). The script imports it
into a throw-away memory next to OUT.json through the real path (code_import.run → vi_code_sync), then asks every question:
- clingo: code_facts.bridge(repo) + the question's rules through logic._run (objects(…) → vi__tag, the server's path)
- search: memory.recall(query, or_fallback=True) over the code pieces only (fact objects excluded), top 4 (what the VI shows) and top 10
- truth: the definitions below — each a different method from the importer's (line regex, indentation, git diff, Python imports)
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WRITE = r"(?:INSERT\s+(?:OR\s+[A-Z]+\s+)?INTO|REPLACE\s+INTO|(?<!DO )UPDATE|DELETE\s+FROM)\s+{t}\b"


# ── ground truth helpers (independent of vi_mcp.code_facts) ───────────────────────────────────────────────────────────────────────

def py_files(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "*.py"], capture_output=True, text=True, check=True).stdout.split()
    return sorted(p for p in out if not any(x in p for x in ("/.venv/", "node_modules/")))


def modname(p: str) -> str:
    m = p[:-3].replace("/", ".")
    return m[:-9] if m.endswith(".__init__") else m


def spans(src: str, mod: str) -> list[tuple[int, int, str]]:
    """(first line, last line, qualified name) of top-level functions and methods of top-level classes."""
    out = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((min([n.lineno] + [d.lineno for d in n.decorator_list]), n.end_lineno, f"{mod}.{n.name}"))
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((min([m.lineno] + [d.lineno for d in m.decorator_list]), m.end_lineno, f"{mod}.{n.name}.{m.name}"))
    return out


def at(sp, line: int) -> str | None:
    return next((q for a, b, q in sp if a <= line <= b), None)


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.files = {p: (path / p).read_text(encoding="utf-8", errors="replace") for p in py_files(path)}
        self.spans = {p: spans(s, modname(p)) for p, s in self.files.items()}

    def lines(self, regex: str):
        """(path, line number, text, enclosing function) of every line matching regex, comments stripped."""
        r = re.compile(regex)
        for p, s in self.files.items():
            for i, line in enumerate(s.splitlines(), 1):
                code = line.split("#", 1)[0] if "#" in line and not re.search(r"['\"][^'\"]*#", line) else line
                if r.search(code):
                    yield p, i, line, at(self.spans[p], i)

    def aliases(self, path: str, module: str) -> tuple[set[str], set[str]]:
        """(names bound to the module, names imported from it) in one file — by regex over its import lines."""
        pkg, name = module.rsplit(".", 1)
        here = modname(path)
        here_pkg = here if path.endswith("__init__.py") else here.rpartition(".")[0]
        mods, direct = set(), set()
        for line in self.files[path].splitlines():
            s = line.split("#", 1)[0].strip()  # `from vi_mcp import memory as g  # noqa: E402` (the first run missed these)
            m = re.match(rf"from (\.|{re.escape(pkg)}) import (.+)$", s)
            if m and (m.group(1) == pkg or here_pkg == pkg):
                for part in m.group(2).strip("()").split(","):
                    bits = part.split()
                    if bits and bits[0] == name:
                        mods.add(bits[2] if len(bits) == 3 else name)
            m = re.match(rf"import {re.escape(module)} as (\w+)$", s)
            if m:
                mods.add(m.group(1))
            m = re.match(rf"from (\.{re.escape(name)}|{re.escape(module)}) import (.+)$", s)
            if m and (m.group(1) == module or here_pkg == pkg):
                for part in m.group(2).strip("()").split(","):
                    bits = part.split()
                    if bits:
                        direct.add(bits[2] if len(bits) == 3 else bits[0])
        return mods, direct

    def callers(self, target: str, tests_only: bool = False) -> set[str]:
        module, fn = target.rsplit(".", 1)
        out = set()
        for p, s in self.files.items():
            mods, direct = self.aliases(p, module)
            pats = [rf"\b{re.escape(a)}\.{fn}\(" for a in mods]
            if fn in direct or modname(p) == module:
                pats.append(rf"(?<![\w.]){fn}\(")
            if not pats:
                continue
            r = re.compile("|".join(pats))
            for i, line in enumerate(s.splitlines(), 1):
                if r.search(line) and not line.lstrip().startswith(("def ", "#")):
                    q = at(self.spans[p], i)
                    if q and (not tests_only or (p.startswith("tests/") and q.rsplit(".", 1)[-1].startswith("test"))):
                        out.add(q)
        return out

    def writers(self, table: str) -> set[str]:
        return {q for _p, _i, _l, q in self.lines(WRITE.format(t=table)) if q}

    def unlocked_writers(self) -> set[str]:
        """Production functions with SQL writing a table on a line that no open `with …lock` block encloses (by indentation)."""
        out = set()
        for p, i, line, q in self.lines(WRITE.format(t=r"[A-Za-z_]\w*")):
            if not q or p.startswith("tests/"):
                continue
            a = next(x for x, b, n in self.spans[p] if n == q)
            src = self.files[p].splitlines()
            ind = len(line) - len(line.lstrip())
            locked = False
            for k in range(a, i):
                w = src[k - 1]
                wi = len(w) - len(w.lstrip())
                if re.match(r"\s*(async\s+)?with\b.*\block\b.*:", w) and wi < ind:
                    closed = any(x.strip() and not x.strip().startswith("#") and len(x) - len(x.lstrip()) <= wi
                                 for x in src[k:i - 1])
                    locked = locked or not closed
            if not locked:
                out.add(q)
        return out

    def raisers(self, exc: str) -> set[str]:
        return {q for _p, _i, _l, q in self.lines(rf"\braise\s+{exc}\b") if q}

    def import_time(self) -> dict[str, set[str]]:
        """module → repository modules it imports outside any function (ast walk that skips function bodies)."""
        mods = {modname(p) for p in self.files}
        out = {}
        for p, s in self.files.items():
            me = modname(p)
            pkg = me if p.endswith("__init__.py") else me.rpartition(".")[0]
            got = set()

            def walk(nodes):
                for n in nodes:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                        continue
                    if isinstance(n, ast.If) and "TYPE_CHECKING" in ast.unparse(n.test):
                        continue
                    if isinstance(n, ast.Import):
                        got.update(a.name for a in n.names)
                    elif isinstance(n, ast.ImportFrom):
                        base = n.module or ""
                        if n.level:
                            parts = pkg.split(".")[: len(pkg.split(".")) - (n.level - 1)]
                            base = ".".join(parts + ([n.module] if n.module else []))
                        for a in n.names:
                            got.add(f"{base}.{a.name}" if f"{base}.{a.name}" in mods else base)
                    walk(list(ast.iter_child_nodes(n)))
            walk(ast.parse(s).body)
            out[me] = {m for m in got if m in mods}
        return out

    def cycles(self) -> set[str]:
        g = self.import_time()
        out = set()
        for start in g:
            seen, todo = set(), list(g[start])
            while todo:
                m = todo.pop()
                if m == start:
                    out.add(start)
                    break
                if m not in seen:
                    seen.add(m)
                    todo += list(g.get(m, ()))
        return out

    def duplicates(self, min_nodes: int = 12) -> set[str]:
        by = {}
        for p, s in self.files.items():
            if p.startswith("tests/"):
                continue
            for n in ast.walk(ast.parse(s)):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = [x for x in n.body if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str))]
                    if sum(1 for x in body for _ in ast.walk(x)) < min_nodes:
                        continue
                    key = "\n".join(ast.dump(x) for x in body)
                    q = at(self.spans[p], n.lineno)
                    if q and q.endswith("." + n.name):
                        by.setdefault(key, set()).add((modname(p), q))
        return {q for v in by.values() if len({m for m, _ in v}) > 1 for _m, q in v}

    def changed_in_head(self) -> set[str]:
        diff = subprocess.run(["git", "-C", str(self.path), "diff", "-U0", "HEAD~1", "HEAD", "--", "*.py"], capture_output=True, text=True,
                              check=True).stdout
        out, path = set(), None
        for line in diff.splitlines():
            if line.startswith("+++ "):
                path = line[6:] if line.startswith("+++ b/") else None
            m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if m and path in self.spans:
                a, n = int(m.group(1)), int(m.group(2) or 1)
                out |= {q for k in range(a, a + n) for q in [at(self.spans[path], k)] if q}
        return out


def dangling_aliases(repo: Path) -> set[str]:
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from vi_mcp import api_en, tools; "
            "print('\\n'.join(sorted(set(api_en.TOOL_ALIASES.values()) - {t['name'] for t in tools.TOOLS})))")
    out = subprocess.run([sys.executable, "-c", code, str(repo)], capture_output=True, text=True, check=True).stdout
    return set(out.split())


# ── the questions ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

TM = "tm(M) <= in_module(F, M) & test(F)\n"


def questions(t: Repo, head12: str) -> list[dict]:
    return [
        {"id": "Q01", "q": "Which functions write table object?", "truth": t.writers("object"),
         "rules": 'ans(F) <= writes_table(F, "object")', "search": "INSERT INTO object"},
        {"id": "Q02", "q": "Which functions write table edge?", "truth": t.writers("edge"),
         "rules": 'ans(F) <= writes_table(F, "edge")', "search": "INSERT INTO edge"},
        {"id": "Q03", "q": "Which functions write table tag?", "truth": t.writers("tag"),
         "rules": 'ans(F) <= writes_table(F, "tag")', "search": "INSERT INTO tag"},
        {"id": "Q04", "q": "What calls memory.save?", "truth": t.callers("vi_mcp.memory.save"),
         "rules": 'ans(F) <= calls(F, "vi_mcp.memory.save")', "search": "save"},
        {"id": "Q05", "q": "What calls memory.new_object?", "truth": t.callers("vi_mcp.memory.new_object"),
         "rules": 'ans(F) <= calls(F, "vi_mcp.memory.new_object")', "search": "new_object"},
        {"id": "Q06", "q": "What calls logic.check?", "truth": t.callers("vi_mcp.logic.check"),
         "rules": 'ans(F) <= calls(F, "vi_mcp.logic.check")', "search": "logic.check"},
        {"id": "Q07", "q": "What calls code_import.server_sync?", "truth": t.callers("vi_mcp.code_import.server_sync"),
         "rules": 'ans(F) <= calls(F, "vi_mcp.code_import.server_sync")', "search": "server_sync"},
        {"id": "Q08", "q": "Which functions raise LogicError?", "truth": t.raisers("LogicError"),
         "rules": 'ans(F) <= raises(F, "clingo_object.notation.LogicError")', "search": "raise LogicError"},
        {"id": "Q09", "q": "Which functions of vi_mcp.tools raise Rejected?",
         "truth": {q for q in t.raisers("Rejected") if q.startswith("vi_mcp.tools.")},
         "rules": 'ans(F) <= raises(F, "vi_mcp.tools.Rejected") & in_module(F, "vi_mcp.tools")', "search": "raise Rejected"},
        {"id": "Q10", "q": "Which modules import vi_mcp.memory when they are loaded?",
         "truth": {m for m, s in t.import_time().items() if "vi_mcp.memory" in s},
         "rules": 'ans(M) <= imports(M, "vi_mcp.memory")', "search": "from . import memory", "modules": True},
        {"id": "Q11", "q": "Which repository modules does clingo_object.notation import when it is loaded?",
         "truth": t.import_time()["clingo_object.notation"],
         "rules": 'ans(M) <= imports("clingo_object.notation", M) & module(M)', "search": None},
        {"id": "Q12", "q": "Which modules import clingo_object.notation when they are loaded?",
         "truth": {m for m, s in t.import_time().items() if "clingo_object.notation" in s},
         "rules": 'ans(M) <= imports(M, "clingo_object.notation")', "search": "clingo_object.notation import", "modules": True},
        {"id": "Q13", "q": "Is there an import cycle (which modules are on one)?", "truth": t.cycles(),
         "rules": "reach(A, B) <= imports(A, B) & module(B)\nreach(A, C) <= reach(A, B) & imports(B, C) & module(C)\nans(A) <= reach(A, A)",
         "search": None},
        {"id": "Q14", "q": "Which tests call code_import.run?", "truth": t.callers("vi_mcp.code_import.run", tests_only=True),
         "rules": 'ans(T) <= calls(T, "vi_mcp.code_import.run") & test(T)', "search": "ci.run"},
        {"id": "Q15", "q": "Which tests call memory.save directly?", "truth": t.callers("vi_mcp.memory.save", tests_only=True),
         "rules": 'ans(T) <= calls(T, "vi_mcp.memory.save") & test(T)', "search": "g.save test"},
        {"id": "Q16", "q": "Which production functions write a table outside a lock?", "truth": t.unlocked_writers(),
         "rules": TM + "ans(F) <= writes_unlocked(F, T) & in_module(F, M) & ~tm(M)", "search": "INSERT INTO"},
        {"id": "Q17", "q": "Which production functions have the same body as one in another module?", "truth": t.duplicates(),
         "rules": TM + "ans(F) <= body(F, H) & body(G, H) & in_module(F, M) & in_module(G, N) & (M != N) & ~tm(M) & ~tm(N)",
         "search": None},
        {"id": "Q18", "q": "Which functions did the newest commit change?", "truth": t.changed_in_head(),
         "rules": f'ans(F) <= changed_in(F, "{head12}")', "search": None},
        {"id": "Q19", "q": "Which tool aliases point at no registered tool?", "truth": dangling_aliases(t.path),
         "rules": 'reg(N) <= const_value("vi_mcp.tools.TOOLS", N)\nans(E) <= const_value("vi_mcp.api_en.TOOL_ALIASES", E) & ~reg(E)',
         "search": None},
        {"id": "Q20", "q": "Which functions call memory.save while holding the lock?",
         "truth": None, "rules": 'ans(F) <= calls_locked(F, "vi_mcp.memory.save")', "search": "with g.lock save"},
    ]


def lock_truth(t: Repo) -> set[str]:
    """Q20: callers of memory.save whose call line sits in an open `with …lock` block (indentation, as unlocked_writers)."""
    out = set()
    for p, s in t.files.items():
        mods, direct = t.aliases(p, "vi_mcp.memory")
        pats = [rf"\b{re.escape(a)}\.save\(" for a in mods] + ([r"(?<![\w.])save\("] if p == "vi_mcp/memory.py" or "save" in direct else [])
        if not pats:
            continue
        src = s.splitlines()
        for i, line in enumerate(src, 1):
            if not re.search("|".join(pats), line) or line.lstrip().startswith("def "):
                continue
            q = at(t.spans[p], i)
            if not q:
                continue
            a = next(x for x, b, n in t.spans[p] if n == q)
            ind = len(line) - len(line.lstrip())
            for k in range(a, i):
                w = src[k - 1]
                wi = len(w) - len(w.lstrip())
                if re.match(r"\s*(async\s+)?with\b.*\block\b.*:", w) and wi < ind and not any(
                        x.strip() and len(x) - len(x.lstrip()) <= wi for x in src[k:i - 1]):
                    out.add(q)
    return out


# ── running ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def piece_names(o) -> set[str]:
    """The function names a code piece stands for: module.symbol; a class piece stands for all its methods ('Class.*')."""
    path, sym = o.data.get("datei", ""), str(o.data.get("symbol", ""))
    if not path.endswith(".py"):
        return set()
    if sym == "(module)":
        return {"MODULE:" + modname(path)}
    return {f"{modname(path)}.{sym}"}


def score(answer: set[str], truth: set[str]) -> dict:
    hit = answer & truth
    return {"n": len(answer), "hit": len(hit), "precision": round(len(hit) / len(answer), 3) if answer else (1.0 if not truth else 0.0),
            "recall": round(len(hit) / len(truth), 3) if truth else (1.0 if not answer else 0.0), "exact": answer == truth}


def main(subject: Path, out: Path) -> int:
    from vi_mcp import code_facts, code_import, logic, session, tools
    from vi_mcp import memory as g
    out.parent.mkdir(parents=True, exist_ok=True)
    db = out.parent / "measure-vi.db"
    for f in out.parent.glob("measure-vi.db*"):
        f.unlink()
    g.open_db(db)
    session.end_all("unterbrochen")
    t0 = time.monotonic()
    imp = code_import.run(subject, "vi-mcp", lambda tool, args: tools.call_tool(tool, args, "measure", False), name="vi-mcp")
    import_s = time.monotonic() - t0
    head12 = subprocess.run(["git", "-C", str(subject), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
    t = Repo(subject)
    qs = questions(t, head12)
    qs[-1]["truth"] = lock_truth(t)
    bridge = code_facts.bridge("vi-mcp")
    rows = []
    for q in qs:
        t1 = time.monotonic()
        r = logic._run(bridge + q["rules"] + "\n")
        clingo_s = time.monotonic() - t1
        models = [set(m) for m in r["models"]]
        atoms = set.intersection(*models) if models else set()
        ans = {a[4:-1].strip('"') for a in atoms if a.startswith("ans(")}
        truth = set(q["truth"])
        row = {"id": q["id"], "question": q["q"], "truth_n": len(truth), "clingo": score(ans, truth), "clingo_s": round(clingo_s, 2),
               "clingo_missing": sorted(truth - ans)[:10], "clingo_extra": sorted(ans - truth)[:10]}
        if q["search"]:
            hits = g.recall(q["search"], limit=500, viewer=None, or_fallback=True)
            pieces = [o for _s, o, _d in hits if "art_wissen:code" in o.hashtags and not code_import._is_fact(o)]
            for k in (4, 10):
                names = set().union(*[piece_names(o) for o in pieces[:k]]) if pieces else set()
                if q.get("modules"):
                    names = {n[7:] for n in names if n.startswith("MODULE:")}
                row[f"search_top{k}"] = score(names, truth)
        else:
            row["search_top4"] = row["search_top10"] = "no answer (the question is not a text query)"
        rows.append(row)
    summary = {"subject_commit": head12, "import": imp, "import_s": round(import_s, 1), "questions": len(rows),
               "clingo_exact": sum(r["clingo"]["exact"] for r in rows),
               "search_top4_exact": sum(isinstance(r["search_top4"], dict) and r["search_top4"]["exact"] for r in rows),
               "search_top10_exact": sum(isinstance(r["search_top10"], dict) and r["search_top10"]["exact"] for r in rows),
               "search_answerable": sum(isinstance(r["search_top4"], dict) for r in rows)}
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    for r in rows:
        s4 = r["search_top4"]
        print(f"{r['id']} truth {r['truth_n']:>3}  clingo {r['clingo']['hit']}/{r['clingo']['n']} exact={r['clingo']['exact']}  "
              + (f"search@4 {s4['hit']}/{s4['n']} R={s4['recall']}  @10 R={r['search_top10']['recall']}" if isinstance(s4, dict) else "search n/a"))
    g.close_db()
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
