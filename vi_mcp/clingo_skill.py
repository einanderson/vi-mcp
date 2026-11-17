"""The clingo skill (0.69.4): how an AI writes, checks and reads logic programs with vi_logic — clingo through the notation clingo-object.

The user (17.11.2026): "Gibt es bereits einen Skill clingo/clingo-object in dem MCP-Server? Wenn nein, dann mache einen, damit eine KI
clingo kann. Solche Informationen müssen vererbt werden." There was none: only the long vi_logic tool description. This text is the one
source of the initial skill clingo: every server writes it into its memory at start as `insight:skill-clingo` (knowledge_kind:skill,
vi_mcp/knowledge_skills.py); docs/CLINGO_SKILL.md is generated from it with `vi-mcp logic skill`.

Written for small instruction-following models: numbered steps, one thing per line, every example with its real result. Every example
in EXAMPLES runs through the real vi_logic tool in tests/test_clingo_skill.py — a lesson that the engine contradicts fails the build.
The pitfalls are the ones measured on 17.11.2026 against clingo 5.8.3 (two of them are silent: no error, just no result).
"""
from __future__ import annotations

#: every example: the vi_logic arguments and the lines its answer must contain (checked against the real engine by the test)
EXAMPLES = [
    {"name": "facts-and-a-rule", "why": "a rule derives new facts",
     "args": {"action": "check", "source": 'zone("gate1", "dmz")\nzone("web1", "lan")\nplanned(run("patch", "gate1"))\n'
                                           'forbidden(run(R, H)) <= zone(H, "dmz") & planned(run(R, H))'},
     "expect": ["state: satisfied", 'forbidden(run("patch","gate1"))']},
    {"name": "a-constraint", "why": "never(...) reports a violation instead of failing",
     "args": {"action": "check", "source": 'host("a")\nhost("b")\nbackup("a")\nnever(host(H) & ~backup(H))'},
     "expect": ["state: violated", "violated never #4 (line 4): never(host(H) & ~backup(H))"]},
    {"name": "a-comparison", "why": "comparisons go in parentheses",
     "args": {"action": "check", "source": 'load("a", 5)\nload("b", 90)\nhot(H) <= load(H, N) & (N > 80)'},
     "expect": ['hot("b")']},
    {"name": "a-norm-conflict", "why": "obligatory and forbidden together is a named conflict",
     "args": {"action": "check", "source": 'obligatory(review("c1"))\nforbidden(review("c1"))'},
     "expect": ["state: violated", 'norm conflict: conflict(norm,review("c1"))']},
    {"name": "a-duty-not-done", "why": "an obligation without done(...) derives violation(...)",
     "args": {"action": "check", "source": 'obligatory(backup("db1"))'},
     "expect": ['violation(backup("db1"))']},
    {"name": "a-contradiction", "why": "p and -p together is reported, never silently empty",
     "args": {"action": "check", "source": 'up("web1")\n-up("web1")'},
     "expect": ["state: contradiction", 'both: up("web1")']},
    {"name": "asking", "why": "ask gives one of six answers per atom",
     "args": {"action": "ask", "source": 'up("web1")\n-up("db1")\nmonitored(H) <= up(H)',
              "queries": ['up("web1")', 'up("db1")', 'up("mail1")', 'down("web1")', 'monitored("web1")']},
     "expect": ['up("web1"): true', 'up("db1"): false', 'up("mail1"): not_derived', 'down("web1"): unknown', 'monitored("web1"): true']},
    {"name": "two-answer-sets", "why": "when answer sets disagree the answer is undetermined",
     "args": {"action": "ask", "source": "a <= ~b\nb <= ~a", "queries": ["a"]},
     "expect": ["a: undetermined"]},
    {"name": "a-soft-rule", "why": "soft(rule, p) may be broken at a cost; probabilities with probabilities=true",
     "args": {"action": "ask", "source": 'soft(reliable(S) <= source(S), 0.8)\nsource("doc_a")', "queries": ['reliable("doc_a")'],
              "probabilities": True},
     "expect": ['reliable("doc_a"): true · P = 0.8']},
]

#: measured mistakes: what the model writes, what happens, what to write instead
PITFALLS = [
    ('p(X) :- q(X).', 'error "illegal target for annotation"', "p(X) <= q(X) — the notation is Python, not raw ASP"),
    ('q("a").', 'error "invalid syntax"', 'q("a") — no full stop at the end of a line'),
    ('p(X) <= q(X) and r(X)', 'error "expected a predicate"', "p(X) <= q(X) & r(X) — & means and"),
    ('p(X) <= q(X) & not r(X)', 'error "invalid syntax"', "p(X) <= q(X) & ~r(X) — ~ means not derivable"),
    ('big(X) <= size(X, N) & N > 5', 'error "a comparison inside a rule goes in parentheses"', "big(X) <= size(X, N) & (N > 5)"),
    ('big(x) <= size(x, n)', "SILENT: no error, nothing follows — lower-case x and n are constants, not variables",
     "big(X) <= size(X, N) — variables start with a capital letter"),
    ('zone(gate1, dmz) and later zone("gate1", "dmz")', "SILENT: gate1 and \"gate1\" are different values, the rule never matches",
     'quote every string value, always: zone("gate1", "dmz")'),
    ('p(X) <= ~q(X)', 'error "unsafe variables"', "every variable must also occur in a positive part of the body: p(X) <= host(X) & ~q(X)"),
    ('n(C) <= (C == count(p(X)))', 'error "unsafe variables" — there is no count/sum/min/max', "list the cases as facts, or count outside"),
    ('a | b', 'error "expected a predicate"', "no disjunction in the head: write two rules with ~ (a <= ~b, b <= ~a)"),
]


def _example_block(e: dict) -> str:
    a = e["args"]
    call = f'vi_logic(action="{a["action"]}"' + (f', queries={a["queries"]}' if "queries" in a else "") + \
           (", probabilities=true" if a.get("probabilities") else "") + ", source=<below>)"
    return (f"#### {e['name']} — {e['why']}\n\n```\n{a['source']}\n```\n\n{call} answers (among others):\n\n```\n"
            + "\n".join(e["expect"]) + "\n```\n")


SKILL_HEAD = """# Skill: clingo — logic programs with vi_logic (clingo-object notation)

You can decide questions of logic with the tool `vi_logic`: what follows from facts and rules, what is allowed, forbidden or required,
whether rules contradict each other, and which memory objects match a condition. The server translates your program to answer set
programming and solves it with clingo 5.8.3. You never write raw clingo/ASP; you write a Python-like notation, and the server never runs
Python. Everything the engine derives is evidence `inferred`, never `measured` — say so when you report it.

## When to use it — and when not

Use vi_logic when the answer must follow from rules, not from your impression:
1. Is X allowed, forbidden or required under these rules? (norms)
2. Do these rules or facts contradict each other? (consistency)
3. What follows if these facts hold? (derivation)
4. Which memory objects match a condition with several parts? (the object dialect, section 6)
Do NOT use it for arithmetic over many numbers, counting, text search or ranking: there are no aggregates (no count, sum, min, max).
Use vi_recall for search.

## 1. The notation, one line per statement

| What | Write | Meaning |
|---|---|---|
| fact | `zone("gate1", "dmz")` | lower-case predicate, values in double quotes (or numbers) |
| rule | `in_dmz(H) <= zone(H, "dmz")` | head `<=` body; the head holds when the body holds |
| and | `a(X) <= b(X) & c(X)` | `&` joins conditions |
| variable | `X`, `Host`, `N` | starts with a CAPITAL letter; the same name means the same value in one rule |
| not derivable | `~done(A)` | true when done(A) cannot be derived (negation as failure) |
| explicitly false | `-up("db1")` | strong negation: up("db1") is false, not just unknown |
| comparison | `(N > 80)`, `(H != "gate1")`, `(X == "a")` | ALWAYS in parentheses |
| constraint | `never(host(H) & ~backup(H))` | a forbidden combination; a match is REPORTED as a violation |
| norm | `obligatory(A)`, `forbidden(A)`, `permitted(A)`, `done(A)` | built in, see section 4 |
| soft rule | `soft(head <= body, 0.8)` | may be broken at a cost; 0.8 is the prior probability, see section 5 |

Nested terms are fine: `planned(run("patch", "gate1"))`. One statement per line. No full stop at the end. Comments with `#`.

## 2. The workflow — do it in this order

1. Write the facts you know, each with quoted values.
2. Write the rules. Check every rule: does each variable of the head and of every `~`, `-` or comparison also occur in a positive
   part of the body? If not, the program is unsafe.
3. Call `vi_logic(action="check", source=...)`.
4. Read the first line: `state: satisfied`, `state: violated`, `state: contradiction`, `state: timeout` or `state: unsatisfiable`.
5. If you get an error, fix exactly what it names (section 7 lists the measured errors) and call check again.
6. To answer a yes/no question about one atom, call `vi_logic(action="ask", source=..., queries=["atom(...)", ...])`.
7. Report the answer with its state and the words "inferred by vi_logic". Never turn `unknown`, `not_derived` or `undetermined` into
   yes or no.
8. Keep a rule set you will need again: `vi_logic(action="save", source=..., name="my-rules", description=...)`; later
   `rule_set="my-rules"` in any action adds it in front of your source.

## 3. The six answers of ask

| Answer | Means | What you say |
|---|---|---|
| `true` | follows in every answer set | "yes (inferred)" |
| `false` | the explicit negation `-atom` follows | "no (inferred)" |
| `both` | atom and -atom both follow | "contradiction in the rules or facts" — name both sides |
| `not_derived` | the predicate exists in the program, this instance does not follow | "does not follow from what is known" — NOT "no" |
| `unknown` | the program says nothing about this predicate | "the rules do not cover this" |
| `undetermined` | answer sets disagree | "depends on a choice the rules leave open" |

`false` needs an explicit `-atom`. `~atom` in a rule never makes an atom false; it only reads "cannot be derived".

## 4. Norms (the deontic square, built in)

- `obligatory(A)` without `done(A)` derives `violation(A)`.
- `obligatory(A)` and `forbidden(A)` together is a named conflict: `norm conflict: conflict(norm,A)`; the state is `violated`.
- `obligatory(A)` derives `permitted(A)`.
- Mark what was done as a fact: `done(backup("db1"))`.

## 5. Soft rules and learned weights

- `soft(reliable(S) <= source(S), 0.8)`: the rule holds with prior probability 0.8. With `probabilities=true`, ask returns `P = …`.
- When you have checked a conclusion against reality, call `vi_logic(action="rate", source=..., conclusion="reliable(\\"doc_a\\")",
  correct=true|false)`: the rules that derived it learn their weight. A hard rule stays hard; its measurement is only shown.

## 6. The object dialect: rules over the memory's objects

- `(O in objects("kind:insight host:web*"))` binds O to every object a vi_recall filter finds — only objects you may see; objects in
  quarantine or the archive are never imported.
- `(O.host == H)` reads a hashtag (host:…); `(O.kind == "insight")` reads a field: kind, mode, visibility, instance, index, text; numbers
  age (days), importance, confirmed, refuted, accesses, e.g. `(O.age > 90)`.
- `linked(O, B)` follows an edge of the memory.
- Example (tests/test_logic_objects.py):
  `in_dmz(H) <= (O in objects("zone:dmz")) & (O.host == H)` and `never(planned(R, H) & in_dmz(H))`.
- `propose(tag(O, "status", "blocked"))` or `propose(link(A, B, "reason"))` as a head: `vi_logic(action="propose", …)` holds the
  proposal for review; someone other than its author decides it.
- `vi_logic(action="objects", filters=[...])` returns the objects as JSON (data for the Python package below).
- `vi_logic(action="answer", question=...)` lets the server's model write the program for a question in natural language — read the
  program it shows before you trust the answer.

## 7. Measured mistakes (clingo 5.8.3, 17.11.2026)

| You wrote | What happens | Write instead |
|---|---|---|
"""

SKILL_TAIL = """
## 8. Checking an answer independently

`vi_logic(action="verify", source=..., expected_answers=[...], reference=..., question=..., predicate="ans")` compares what your program
derives for `ans(...)` with a complete answer set you obtained independently (a test, a query, a person). Use it before you call a
program right: a program that runs is not yet a program that means the right thing.

## 9. On a trusted machine: the Python package clingo-object

The same language as real Python classes (only where code is trusted — an agent's own machine, tests; never on the server):

```python
from clingo_object import VIObject, Program, Obj, rel

class Knowledge(VIObject):
    def stale(self):
        return (self.mode == "test") & (self.age > 90)

O = Obj("O")
program = Program(Knowledge, rel.review(O) <= O.where("topic:oo") & O.stale())
print(program.notation())   # the notation the server runs
```

`Memory` (clingo_object.client) fetches the objects through `vi_logic action=objects`. Philosophy packages (a checked rule set an AI
works by) are the skill `philosophie` / `vi_philosophy`.

## 10. Checklist before you report

- [ ] every variable starts with a capital letter, every string value is quoted
- [ ] every comparison is in parentheses, every line has no full stop
- [ ] check returned `state: satisfied` or `state: violated` (not an error, not `contradiction` unless that is the finding)
- [ ] `unknown`, `not_derived` and `undetermined` are reported as such, never as no
- [ ] the report says "inferred by vi_logic" and names the rules or the saved rule set
"""


def skill() -> str:
    """The whole skill text, examples and pitfalls included."""
    pit = "\n".join(f"| `{w}` | {h} | {f} |" for w, h, f in PITFALLS)
    ex = "\n## Examples (each one checked against the engine by the test suite)\n\n" + "\n".join(_example_block(e) for e in EXAMPLES)
    return SKILL_HEAD + pit + "\n" + ex + SKILL_TAIL


SKILL_FILE_HEAD = """---
name: clingo
description: Write, check and read logic programs with vi_logic (clingo-object notation, clingo 5.8.3) — what is allowed, forbidden or required, consistency, derivations, rules over memory objects. Use when a question must be decided by rules, not by impression.
---
<!-- Generated from vi-mcp vi_mcp/clingo_skill.py (the one source). Regenerate: vi-mcp logic skill --file > SKILL.md. Tool: vi_logic(action="skill"). -->
"""
