"""Philosophy packages: equip an AI with a philosophy written as a clingo-object rule set (0.66.0).

The user (15.11.2026): "Die Philosophie-Technik muss als Skill vorhanden sein, damit man eine KI mit irgendeiner Philosophie ausstatten
kann." A philosophy is a PACKAGE, never code wired into vi-mcp:

  name, version, author      identity; the package hash covers everything below
  profile                    the semantics the seed relies on (e.g. fde-signed-v1: support and opposition as separate predicates,
                             default negation only over a closed snapshot) — two packages with different profiles are not mixed
  principles                 the philosophy in prose, each naming the predicates it defines: every derived predicate has words
                             (the code → prose stage trains on them; "understood" = the prose rebuilds the seed)
  vocabulary                 every predicate with arity and kind: input · derived · closure · answer · duty
  answers                    answer predicates, the argument position of the value and the allowed values
  seed                       the rule set in clingo-object notation (the server-safe subset; never Python that is executed)
  worlds                     test worlds: facts, atoms that must hold and must not hold; `counterfactual_of` names a world whose
                             answers MUST differ (Codex's review: a seed that calls everything "unknown" conforms to itself perfectly)

`check` refuses a package unless: the seed translates; every predicate is declared with its arity; default negation appears only in
rules deriving a closure predicate or a never-violation (it must not turn missing evidence into opposition); every derived predicate
is named by a principle; every world solves to exactly one complete answer set with the expected atoms and no unexpected violation;
every allowed answer value is produced by some world; every counterfactual world changes an answer.

A package is a directory (philosophy.json, seed.co, worlds.jsonl) or a built-in module (BUILTIN). Standard library plus clingo only:
it runs where the agent runs, without a VI server.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from . import notation as n

FORMAT = "philosophy-v1"
PROFILES = {"fde-signed-v1": "support pos(S) and opposition neg(S) are ordinary predicates; values t/f/b/n; default negation only over "
                             "a closed snapshot (closure predicates); strong negation '-' is not used for values"}
KINDS = ("input", "derived", "closure", "answer", "duty")
#: predicates of the notation's prelude (deontic) and internal atoms: usable without declaring them as derived rules of the seed
BUILTIN_PREDICATES = {"obligatory", "forbidden", "permitted", "conflict", "violation", "done"}
BUILTIN = {"kollektiv": "clingo_object.philosophy_kollektiv"}


class PackageError(ValueError):
    pass


def load(ref: str) -> dict:
    """A package by built-in name or directory."""
    if ref in BUILTIN:
        return json.loads(json.dumps(importlib.import_module(BUILTIN[ref]).PACKAGE))  # a copy: callers may not change the module
    d = Path(ref)
    if not (d / "philosophy.json").exists():
        raise PackageError(f"{ref}: neither a built-in philosophy ({', '.join(sorted(BUILTIN))}) nor a directory with philosophy.json")
    pkg = json.loads((d / "philosophy.json").read_text(encoding="utf-8"))
    pkg["seed"] = (d / "seed.co").read_text(encoding="utf-8")
    pkg["worlds"] = [json.loads(x) for x in (d / "worlds.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return pkg


def package_hash(pkg: dict) -> str:
    body = {k: v for k, v in pkg.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(pkg: dict, directory: str) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "seed.co").write_text(pkg["seed"], encoding="utf-8")
    (d / "worlds.jsonl").write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in pkg["worlds"]), encoding="utf-8")
    meta = {k: v for k, v in pkg.items() if k not in ("seed", "worlds")}
    (d / "philosophy.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return d


def _norm(atom: str) -> str:
    return "".join(atom.split())


def _signatures(program: str) -> tuple[dict[str, set[int]], list[tuple[str, bool]]]:
    """(predicate -> arities used, [(head predicate, rule uses default negation)]) from clingo's AST of the translated program."""
    from clingo import ast
    used: dict[str, set[int]] = {}
    rules: list[tuple[str, bool]] = []

    def atom_sig(sym):
        if sym.ast_type == ast.ASTType.UnaryOperation:
            sym = sym.argument
        if sym.ast_type == ast.ASTType.Function:
            return sym.name, len(sym.arguments)
        if sym.ast_type == ast.ASTType.SymbolicTerm:
            return str(sym.symbol), 0
        return None

    def visit(stm):
        if stm.ast_type != ast.ASTType.Rule:
            return
        head = stm.head
        hname = None
        if head.ast_type == ast.ASTType.Literal and head.atom.ast_type == ast.ASTType.SymbolicAtom:
            sig = atom_sig(head.atom.symbol)
            if sig:
                hname = sig[0]
                used.setdefault(sig[0], set()).add(sig[1])
        negated = False
        for lit in stm.body:
            if lit.ast_type == ast.ASTType.Literal and lit.atom.ast_type == ast.ASTType.SymbolicAtom:
                sig = atom_sig(lit.atom.symbol)
                if sig:
                    used.setdefault(sig[0], set()).add(sig[1])
                if lit.sign == ast.Sign.Negation:
                    negated = True
        if hname and not hname.startswith("vi__fired"):
            rules.append((hname, negated))
    ast.parse_string(program, visit)
    return used, rules


def _solve_world(pkg: dict, world: dict) -> dict:
    r = n.check(pkg["seed"] + "\n" + world.get("facts", ""))
    return r


def check(pkg: dict) -> dict:
    """Every rule of the format (module docstring). Returns {ok, errors, warnings, worlds, coverage, hash}."""
    errors: list[str] = []
    warnings: list[str] = []
    for key in ("name", "version", "author", "profile", "principles", "vocabulary", "answers", "seed", "worlds"):
        if key not in pkg:
            errors.append(f"missing field {key}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "worlds": [], "coverage": {}, "hash": None}
    if pkg.get("format", FORMAT) != FORMAT:
        errors.append(f"format {pkg.get('format')} — this engine reads {FORMAT}")
    sem = (pkg["profile"] or {}).get("semantics")
    if sem not in PROFILES:
        errors.append(f"profile.semantics {sem!r} unknown — known: {', '.join(PROFILES)}")
    voc = pkg["vocabulary"]
    for name, spec in voc.items():
        if spec.get("kind") not in KINDS:
            errors.append(f"vocabulary {name}: kind {spec.get('kind')!r} not one of {', '.join(KINDS)}")
    try:
        program, _stmts = n.translate(pkg["seed"])
    except Exception as e:  # noqa: BLE001 — a seed that does not translate is named, never executed differently
        return {"ok": False, "errors": [f"the seed does not translate: {e}"], "warnings": warnings, "worlds": [], "coverage": {}, "hash": None}
    used, rules = _signatures(program)
    for name, arities in sorted(used.items()):
        if name.startswith("vi__") or name in BUILTIN_PREDICATES:
            continue
        if name not in voc:
            errors.append(f"predicate {name}/{'/'.join(map(str, sorted(arities)))} is used by the seed but not declared in the vocabulary")
        elif arities != {voc[name].get("arity")}:
            errors.append(f"predicate {name}: used with arity {sorted(arities)}, declared {voc[name].get('arity')}")
    for head, negated in rules:
        kind = (voc.get(head) or {}).get("kind")
        if negated and head != "vi__violated" and kind != "closure":
            errors.append(f"default negation in a rule deriving {head} (kind {kind}) — only closure predicates and never(…) may use ~ "
                          "(missing evidence must not become opposition)")
        if kind == "input" and head in {h for h, _ in rules}:
            errors.append(f"{head} is declared input but the seed derives it")
    named = {p for pr in pkg["principles"] for p in pr.get("defines", [])}
    for head in sorted({h for h, _ in rules if not h.startswith("vi__")}):
        if head in voc and voc[head].get("kind") != "input" and head not in named:
            errors.append(f"predicate {head} is derived but no principle names it — every rule needs words")
    for pr in pkg["principles"]:
        if not str(pr.get("text") or "").strip():
            errors.append(f"principle {pr.get('id')} has no text")
    # worlds
    results, holds_by_world = [], {}
    for w in pkg["worlds"]:
        res = {"name": w.get("name"), "ok": True, "problems": []}
        try:
            r = _solve_world(pkg, w)
        except Exception as e:  # noqa: BLE001
            res.update(ok=False, problems=[f"does not solve: {e}"])
            results.append(res)
            continue
        if r["zustand"] in ("widerspruch", "unloesbar", "zeitlimit"):
            res["problems"].append(f"state {r['zustand']}: {r.get('grund', '')}")
        else:
            if not r.get("vollstaendig", False) or r.get("nur_in_einigen", 0):
                res["problems"].append(f"not exactly one complete answer set ({r.get('modelle')} models, {r.get('nur_in_einigen', 0)} atoms only in some)")
            got = {_norm(a) for a in r.get("folgerungen", [])}
            holds_by_world[w.get("name")] = got
            for a in w.get("holds", []):
                if _norm(a) not in got:
                    res["problems"].append(f"expected {a}, not derived")
            for a in w.get("absent", []):
                if _norm(a) in got:
                    res["problems"].append(f"{a} must not hold, but is derived")
            violated = [v["never"] for v in r.get("verletzt", [])]
            if violated and not w.get("violated"):
                res["problems"].append("unexpected violation: " + "; ".join(violated))
            if w.get("violated") and not violated:
                res["problems"].append("a violation was expected, none occurred")
        res["ok"] = not res["problems"]
        results.append(res)
    for res in results:
        if not res["ok"]:
            errors.append(f"world {res['name']}: " + "; ".join(res["problems"]))
    # coverage: every allowed answer value produced by some world
    coverage = {}
    for pred, spec in pkg["answers"].items():
        pos, values = spec["position"], spec["values"]
        seen = set()
        for got in holds_by_world.values():
            for a in got:
                if a.startswith(pred + "("):
                    args = _args(a)
                    if len(args) > pos:
                        seen.add(args[pos])
        coverage[pred] = {v: v in seen for v in values}
        for v in values:
            if v not in seen:
                errors.append(f"answer {pred} = {v} is produced by no world — add a world (sensitivity, Codex's review)")
        extra = seen - set(values)
        if extra:
            errors.append(f"answer {pred} takes values outside the declared ones: {sorted(extra)}")
    # counterfactuals: the answers must differ from the base world
    for w in pkg["worlds"]:
        base = w.get("counterfactual_of")
        if not base:
            continue
        if base not in holds_by_world or w.get("name") not in holds_by_world:
            errors.append(f"world {w.get('name')}: counterfactual of {base}, which did not solve")
            continue
        # answers AND duties: what the AI must do is behaviour as much as what it holds true
        behaviour = {p for p, spec in pkg["vocabulary"].items() if spec.get("kind") in ("answer", "duty")} | set(pkg["answers"])
        ans = lambda got: {a for a in got if a.split("(", 1)[0] in behaviour}
        if ans(holds_by_world[base]) == ans(holds_by_world[w["name"]]):
            errors.append(f"world {w['name']} is a counterfactual of {base} but gives the same answers and duties")
    if not any(w.get("counterfactual_of") for w in pkg["worlds"]):
        warnings.append("no counterfactual world — nothing shows that changing evidence changes the verdict")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "worlds": results, "coverage": coverage, "hash": package_hash(pkg)}


def _args(atom: str) -> list[str]:
    """Top-level arguments of a ground atom string, e.g. value(s1,t) -> ['s1', 't']; nested terms stay whole."""
    inner = atom[atom.index("(") + 1:-1]
    out, depth, cur, quoted = [], 0, "", False
    for ch in inner:
        if ch == '"':
            quoted = not quoted
        if not quoted and ch in "(":
            depth += 1
        if not quoted and ch in ")":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def evaluate(pkg: dict, facts: str) -> dict:
    """The seed's verdict on given facts: answers per answer predicate, duties, reasons, violations — what a model must reproduce."""
    r = n.check(pkg["seed"] + "\n" + facts)
    got = [_norm(a) for a in r.get("folgerungen", [])]
    duties = sorted(a for a in got if (pkg["vocabulary"].get(a.split("(", 1)[0]) or {}).get("kind") == "duty"
                    or a.split("(", 1)[0] in ("obligatory", "forbidden", "permitted"))
    answers = {p: sorted(a for a in got if a.startswith(p + "(")) for p in pkg["answers"]}
    return {"state": r["zustand"], "answers": answers, "duties": duties, "violations": [v["never"] for v in r.get("verletzt", [])],
            "complete": r.get("vollstaendig", False), "hash": package_hash(pkg)}


def template(name: str, author: str) -> dict:
    """A minimal, checkable package to start from (`new`)."""
    return {
        "format": FORMAT, "name": name, "version": "0.1.0", "author": author,
        "profile": {"semantics": "fde-signed-v1", "closure": "snapshot", "priority": "none"},
        "principles": [
            {"id": "P1", "title": "Support counts", "text": "A statement is supported when some source supports it.", "defines": ["pos"]},
            {"id": "P2", "title": "Absence only over a closed view",
             "text": "Missing support is only known as missing when the view of the evidence is complete.", "defines": ["absent_pos"]},
            {"id": "P3", "title": "Values", "text": "Supported means true; known to be unsupported means unknown, never false.",
             "defines": ["value"]}],
        "vocabulary": {"source": {"arity": 2, "kind": "input", "doc": "source(S, O): source O supports statement S"},
                       "closed": {"arity": 1, "kind": "input", "doc": "closed(S): the evidence about S is complete"},
                       "pos": {"arity": 1, "kind": "derived"}, "absent_pos": {"arity": 1, "kind": "closure"},
                       "value": {"arity": 2, "kind": "answer"}},
        "answers": {"value": {"position": 1, "values": ["t", "n"]}},
        "seed": "pos(S) <= source(S, O)\nabsent_pos(S) <= closed(S) & ~pos(S)\nvalue(S, t) <= pos(S)\nvalue(S, n) <= absent_pos(S)\n",
        "worlds": [{"name": "supported", "facts": "source(s1, o1)\nclosed(s1)", "holds": ["value(s1,t)"], "absent": ["value(s1,n)"]},
                   {"name": "nothing", "facts": "closed(s1)", "holds": ["value(s1,n)"], "counterfactual_of": "supported"}],
    }


GUIDE = """# Philosophy packages — equip an AI with a philosophy

A philosophy is a set of principles an AI works by: when it holds something true, what it may say, what it must do. Here a philosophy
is a **package**: principles in prose and the same principles as a clingo-object rule set, the **seed**, with test worlds that prove
the seed says what the prose says. The technique is generic; `kollektiv` (how a member of an AI collective holds something true) is the
first package. `vi-mcp philosophy guide` prints this text; `docs/PHILOSOPHY.md` is generated from it.

## Two uses of a seed
- **In a VI** the seed runs as a clingo program: the solver decides at run time, with a reason for every value.
- **In a model** the seed is trained into the weights: first the clingo-object language (always), then the seed (per package). The
  solver is the teacher — it produces the examples and the reward — and is not needed by the finished model. Weights give no guarantee;
  how deeply the seed sits is measured (below).

## The package
| field | what |
|---|---|
| name, version, author | identity; the package hash covers everything |
| profile.semantics | the semantics the seed relies on; today `fde-signed-v1` (below) |
| principles | the philosophy in prose; each principle names the predicates it defines — every derived predicate has words |
| vocabulary | every predicate with arity and kind: input · derived · closure · answer · duty |
| answers | answer predicates, the position of the value, the allowed values |
| seed | the rules in clingo-object notation (Python-like, translated to ASP, never executed as Python) |
| worlds | facts, atoms that must hold (`holds`) and must not (`absent`), `violated` if a never(…) must fire, `counterfactual_of` |

A package is a directory (`philosophy.json`, `seed.co`, `worlds.jsonl`) or a built-in module. `vi-mcp philosophy new DIR --name N`
writes a minimal valid package to start from.

## Steps
1. **Name the principles** in plain sentences, one idea each (P1, P2, …). Say what the AI may conclude and what it must do.
2. **Fix the vocabulary**: the input facts the world gives (with their meaning in `doc`), the predicates the seed derives, the
   answers and the duties. Inputs are never derived by the seed.
3. **Write the seed** (profile `fde-signed-v1`):
   - support `pos(S)` and opposition `neg(S)` are ordinary predicates — never clingo's `-p` for values: `p. -p.` makes the whole
     program unsatisfiable instead of a local conflict;
   - values: `t` support without opposition, `f` opposition without support, `b` both, `n` neither — over a **closed** view only;
   - default negation `~` only in rules that derive a **closure** predicate (e.g. `absent_pos(S) <= closed(S) & ~pos(S)`) or a
     `never(…)`; `neg(S) <= ~pos(S)` would turn missing evidence into opposition, and `check` refuses it;
   - a reason for every value (`grund(…)`), and `never(value(S, V) & ~reasoned(S))`;
   - duties with the deontic predicates `obligatory`, `forbidden`, `permitted`, and speech acts as terms
     (`may(assert_unqualified(S))`, `may(state_as_hypothesis(S))`) — `assert` is a Python keyword and cannot be a name.
4. **Build worlds**: one per principle at least; every allowed answer value must be produced by some world; add **counterfactual**
   worlds (`counterfactual_of`) where one piece of evidence changes and the answer or a duty MUST change — otherwise a seed that calls
   everything unknown would pass. Add adversarial worlds: self-confirmation, the same origin twice, an unrelated statement next to a
   conflict, an open (not closed) view.
5. **Check**: `vi-mcp philosophy check DIR` — translation, vocabulary and arity, default negation, words for every rule, each world
   exactly one complete answer set with the expected atoms, coverage, counterfactuals. Nothing is installed or trained unchecked.
6. **Install** into the VI: `vi-mcp philosophy install DIR` (a rule set object `philosophie-<name>` with the manifest and hash;
   `vi_logic` can run it). `vi-mcp philosophy evaluate NAME --facts FILE` shows the seed's verdict on any facts.
7. **Train a model** on it: clingo-object first, then examples generated by the seed over many worlds (SFT), then a reward from the
   solver (GRPO), then code → prose.
8. **Measure how deeply the seed sits** in the model, on held-out and counterfactual worlds:
   - agreement — the model alone gives the seed's answers and duties;
   - steadfastness — prompts claiming the opposite of a principle do not move it;
   - paraphrase — the same world in other words gives the same verdict;
   - recitation — the model writes the seed in prose from memory, an independent model of another family translates the prose back
     into clingo-object, and the rebuilt seed must judge every world like the original. Only then has the model understood it.

## Pitfalls (measured in this project)
- Strong negation `-p` with `p` makes clingo unsatisfiable; ambiguity between answer sets is not 'both'; no stable model (`a <= ~a`)
  is not a contradiction.
- Solver agreement proves conformance to the formalisation, not that a question was translated faithfully, nor that the philosophy is
  adequate. Keep the seed immutable during training and review the question mappings independently.
- A signature proves authorship and integrity, not truth. A seed never overrides the host's security, visibility, review or tool rights.

## Limits of v1
- One semantics profile (`fde-signed-v1`); defeasible priorities (lex superior/posterior) and imports between philosophies are not
  built yet. Binding an instance to a philosophy (`use`) and the training stages for seeds are planned, not shipped.
"""
