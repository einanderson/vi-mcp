"""The clingo-object language in both directions: random rule sets and their meaning in plain English (0.66.0).

The user (15.11.2026): "Ich dachte eigentlich, dass wir der KI clingo beibringen und dann den Samen. Und aus ihrem Verständnis heraus kann
sie den Samen in Prosa umformulieren wie bei einer Übersetzung." So a model first learns the LANGUAGE — reading rules and writing them —
on thousands of rule sets that share no predicate with any philosophy seed; only then the seed. When it later translates the seed into
prose, it has never seen the seed's prose: the translation comes from understanding clingo, not from memory.

Every generated program is valid clingo-object notation (checked by translating it); every prose text is produced by a deterministic
renderer from the same rule structure, so code and prose say exactly the same. Rule forms cover what seeds use: conjunction, default
negation (`~p` = "p cannot be derived", never "p is false"), comparison (two different), constants, `never` (a forbidden situation),
deontic duties (obligatory, forbidden, may), recursion (reachability).
"""
from __future__ import annotations

import random

from . import notation as n

#: predicates a generated rule may use — none of them occurs in a philosophy package (exclude= removes more at call time)
UNARY = ["server", "backup", "admin", "critical", "reachable", "patched", "approved", "archived", "monitored", "public", "encrypted",
         "deprecated", "healthy", "expired", "trusted", "internal", "tested", "documented", "running", "locked", "vulnerable",
         "replicated", "orphaned", "scheduled", "staged"]
BINARY = {"runs_on": "runs on", "owns": "owns", "depends_on": "depends on", "located_in": "is located in", "reports": "reports",
          "mentions": "mentions", "manages": "manages", "replaces": "replaces", "connects_to": "connects to", "stores": "stores",
          "watches": "watches", "signs": "signs", "uses": "uses", "hosts": "hosts", "links_to": "links to"}
ACTS = {"restart": "restart", "notify": "notify the owner of", "review": "review", "isolate": "isolate", "delete": "delete",
        "escalate": "escalate", "back_up": "back up", "update": "update"}
CONSTANTS = ["low", "high", "green", "red", "open", "done"]
#: ternary relations: (code name, prose with {0} {1} {2})
TERNARY = {"sends": "{0} sends {1} to {2}", "assigns": "{0} assigns {1} to {2}", "rates": "{0} rates {1} as {2}",
           "logs": "{0} logs {1} at {2}", "grants": "{0} grants {1} to {2}"}
MODES = ["up", "down", "pro", "contra", "yes", "no"]  # no Python keywords (the notation is parsed as Python: "for" fails)
VARS = ["X", "Y", "Z"]


def _u(p: str, v: str) -> tuple[str, str]:
    return f"{p}({v})", f"{v} is {p.replace('_', ' ')}"


def _b(r: str, a: str, b: str) -> tuple[str, str]:
    return f"{r}({a}, {b})", f"{a} {BINARY[r]} {b}"


def _rule(rnd: random.Random, unary: list[str], binary: list[str]) -> tuple[str, str]:
    """One rule as (code, prose)."""
    form = rnd.choice(["and", "not", "rel", "two", "never", "duty_must", "duty_forbid", "may", "const", "single",
                       "neq", "const_body", "ternary_two", "duty_two", "closure"])
    h, a, b, c = rnd.sample(unary, 4)
    r, r2 = rnd.sample(binary, 2)
    if form == "neq":  # a variable must differ from another one (like "not the author")
        return (f"{h}(X) <= {r}(X, Y) & {r2}(X, Z) & (Y != Z)",
                f"X is {h.replace('_', ' ')} when X {BINARY[r]} some Y, X {BINARY[r2]} some Z, and Y is not the same as Z.")
    if form == "const_body":  # a constant inside a body literal
        t = rnd.choice(list(TERNARY))
        m = rnd.choice(MODES)
        return (f"{h}(X) <= {t}(X, {m}, Y)",
                f"X is {h.replace('_', ' ')} when " + TERNARY[t].format("X", m, "some Y") + ".")
    if form == "ternary_two":  # two different third arguments for the same first two (like "two independent lines")
        t = rnd.choice(list(TERNARY))
        return (f"{h}(X, M) <= {t}(X, M, F) & {t}(X, M, G) & (F != G)",
                f"{h.replace('_', ' ').capitalize()} holds for X and M when " + TERNARY[t].format("X", "M", "two different parties") + ".")
    if form == "duty_two":  # a duty with two participants, a function term inside the deontic head
        act = rnd.choice(list(ACTS))
        return (f"forbidden({act}(Y, X)) <= {r}(Y, X)",
                f"If Y {BINARY[r]} X, then Y must not {ACTS[act]} X.")
    if form == "closure":  # absence over a closed view, as philosophy seeds use it
        return (f"missing_{a}(X) <= {b}(X) & ~{a}(X)",
                f"For an X that is {b.replace('_', ' ')}, it counts as missing that X is {a.replace('_', ' ')} when that cannot be derived.")
    if form == "single":
        code_b, pr_b = _u(a, "X")
        return f"{h}(X) <= {code_b}", f"Every X that is {a.replace('_', ' ')} is {h.replace('_', ' ')}."
    if form == "and":
        (cb1, pb1), (cb2, pb2) = _u(a, "X"), _u(b, "X")
        return f"{h}(X) <= {cb1} & {cb2}", f"X is {h.replace('_', ' ')} when {pb1} and {pb2}."
    if form == "not":
        (cb1, pb1), (cb2, _p) = _u(a, "X"), _u(b, "X")
        return (f"{h}(X) <= {cb1} & ~{cb2}",
                f"X is {h.replace('_', ' ')} when {pb1} and it cannot be derived that X is {b.replace('_', ' ')} (absence of evidence, not proof of the opposite).")
    if form == "rel":
        (cr, pr), (cu, pu) = _b(r, "X", "Y"), _u(a, "Y")
        return f"{h}(X) <= {cr} & {cu}", f"X is {h.replace('_', ' ')} when {pr} some Y and {pu}."
    if form == "two":
        return (f"{h}(X) <= {r}(X, Y) & {r}(X, Z) & (Y != Z)",
                f"X is {h.replace('_', ' ')} when X {BINARY[r]} at least two different things.")
    if form == "never":
        (cb1, pb1), (cb2, pb2) = _u(a, "X"), _u(b, "X")
        return f"never({cb1} & {cb2})", f"It is a violation if something is both {a.replace('_', ' ')} and {b.replace('_', ' ')}."
    act = rnd.choice(list(ACTS))
    if form == "duty_must":
        cb1, pb1 = _u(a, "X")
        return f"obligatory({act}(X)) <= {cb1}", f"If {pb1}, one must {ACTS[act]} X."
    if form == "duty_forbid":
        cb1, pb1 = _u(a, "X")
        return f"forbidden({act}(X)) <= {cb1}", f"If {pb1}, one must not {ACTS[act]} X."
    if form == "may":
        cb1, pb1 = _u(a, "X")
        return f"may({act}(X)) <= {cb1}", f"If {pb1}, one may {ACTS[act]} X."
    const = rnd.choice(CONSTANTS)
    cb1, pb1 = _u(a, "X")
    return f"level(X, {const}) <= {cb1}", f"The level of X is {const} when {pb1}."


def program(rnd: random.Random, exclude: set[str] = frozenset(), rules: tuple[int, int] = (1, 4)) -> tuple[str, str]:
    """A random rule set as (code, prose); predicates in `exclude` are never used. Recursion is added sometimes."""
    unary = [p for p in UNARY if p not in exclude]
    binary = [p for p in BINARY if p not in exclude]
    out_c, out_p = [], []
    for _ in range(rnd.randint(*rules)):
        c, p = _rule(rnd, unary, binary)
        if c not in out_c:
            out_c.append(c)
            out_p.append(p)
    if rnd.random() < 0.15:
        r = rnd.choice(binary)
        out_c += [f"reach(X, Y) <= {r}(X, Y)", f"reach(X, Z) <= reach(X, Y) & {r}(Y, Z)"]
        out_p.append(f"Y is reachable from X when X {BINARY[r]} Y, directly or through a chain of such steps.")
    code = "\n".join(out_c)
    n.translate(code)  # never a training example that is not valid notation
    return code, " ".join(out_p)


def examples(count: int, seed: int, exclude: set[str] = frozenset(), system: str = "") -> list[dict]:
    """Translation examples in both directions (explain: code → prose, formalize: prose → code)."""
    rnd = random.Random(seed)
    out = []
    for i in range(count):
        code, prose = program(rnd, exclude)
        if rnd.random() < 0.5:
            q, a, task = f"Translate these clingo-object rules into plain English:\n{code}", prose, "explain_rules"
        else:
            q, a, task = f"Write these rules in clingo-object notation:\n{prose}", code, "write_rules"
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        out.append({"messages": msgs, "meta": {"task": task, "world": i, "seed": seed}})
    return out
