# Code structure facts and the `code` philosophy

Code text stays text: `vi-mcp index import-code` stores the code of a repository as pieces per function, class or text block
(`vi_mcp/code_import.py`). From the same `ast` pass the importer also derives the **structure** of every Python file — what it
defines, calls, imports, writes, raises — and stores it as **fact objects**. clingo-object rules read them, so structure questions get
exact answers: "which functions write table objekt", "what calls memory.save", "is there an import cycle".

## Design: facts are hashtags of memory objects

The facts are hashtags on memory objects, read by the existing path (`logic.object_data` → `notation.facts` → `vi__tag`). There is no
second fact store.

Why:
- One data model. Objects already have hashtags, an owning instance, visibility, an index, and an archive with tombstones.
- One import path. A fact table of its own would need its own visibility, incremental sync and deletion rules.
- One syntax. clingo-object already reads a hashtag as `(O.ruft == G)`.

Each fact object carries `art_wissen:code`, like the text pieces. Every place that keeps code out of hints, training data and
network exports (`intuition`, `dataset`, `classify`, `index_transfer`) therefore keeps the facts out too. The vector catch-up skips them
as well: their text is a generated one-line summary.

Objects hold only **local** facts of one file, such as what this function calls. Relations across files (`tested_by`, import cycles)
are derived by rules, so an incremental sync of one changed file never touches another file's objects.

| object | hashtags (stored names are German, like every stored format) |
|---|---|
| one per function and method, `code:funktion` | `funktion:<module.qualname>` `modul:` `datei:` `ruft:<callee>` `ruft_gesperrt:<callee>` `schreibt:<table>` `schreibt_ungesperrt:<table>` `wirft:<exception>` `test:ja` `koerper:<hash>` `geaendert_in:<commit12>` |
| one per file, `code:modul` | `modul:` `datei:` `importiert:<module>` (at import time) `importiert_spaet:<module>` (inside a function) |
| one per literal module constant, `code:konstante` | `konstante:<module.NAME>` `schluessel:<key>` `wert:<value>` (for a list of records: their `name` values) |

The object ids are fixed per index, repository, path and symbol, like the pieces. A removed function's fact object goes to the
archive like its piece.

**Backfill.** `vi_code_sync state` now reports `mit_fakten`, the Python files that already have facts. The client sends each Python
file without facts once more, so an existing index gets its facts without a forced re-import. A server that does not report
`mit_fakten` (older than this change) gets no facts and no re-sends.

## The bridge and the package

`vi_mcp.code_facts.bridge(repo)` is a clingo-object program. It reads the fact objects of one repository as input predicates:
- `snapshot`, `module`, `imports`, `imports_late`, `defines`, `in_module`;
- `calls`, `calls_locked`, `writes_table`, `writes_unlocked`, `raises`, `test`, `body`, `changed_in`;
- `const_key`, `const_value`.

`direct_facts(scan)` gives the same atoms straight from a scan. `tests/test_code_facts.py` checks that both routes give exactly the
same atoms.

The philosophy package **`code`** (`clingo_object/philosophy_code.py`, built with `clingo_object/philosophy.py`) has four rules over
these facts:

| rule | finding |
|---|---|
| `no_import_cycle` | a module reaches itself through imports that run when it is loaded (imports inside functions do not count) |
| `locked_writes` | a production function writes a guarded table outside `with <x>.lock`, and not every caller holds the lock at the call |
| `english_tools` | an alias in an alias table resolves to no registered name, or a registered name is itself an alias |
| `no_duplicates` | the same function body (exact, docstring excluded) in two production modules |

The four rules follow the same conventions:
- Test code is outside their scope. Test code is every function of a module that contains a test.
- Every finding names its reason (`grund`) and is an obligation, `obligatory(fix(Rule, X))`.
- The commits that last touched the function are shown as `last_touched_by`: where to look, not the cause.
- A writer of a guarded table without a direct test gets `obligatory(add_test(F))`.

Default negation appears only in closure predicates over `snapshot(R)`.

**What is particular to one project is a profile of input facts, never a rule.** It names which table needs the lock, which constant
is an alias table of which registry, and each excused finding with its written reason (`exempt(Rule, X, Reason)`). An excused finding
stays visible as `excused`. `PROFILE_VI_MCP` is this repository's profile. Use strings for names, as the importer stores them:
`guarded("objekt")`, not `guarded(objekt)`. A constant never matches a stored value. The first version of the profile had that
mistake, and the lock rule could never fire. `test_the_repository_holds_its_own_architecture_rules` now fails if it comes back.

```bash
vi-mcp philosophy check code                       # 17 worlds, every check
vi-mcp index code-check REPO --profile vi-mcp      # the rules over a local checkout (exit 1 if a rule is violated)
vi-mcp index code-check REPO --facts               # the facts as ASP
```

In the VI, run the bridge, the profile and the seed (or a question) with `vi_logic`, for example
`code_facts.bridge("vi-mcp") + 'ans(F) <= writes_table(F, "objekt")'`.

## Measurement (15.11.2026)

`scripts/measure_code_facts.py SUBJECT OUT.json` asks 20 structure questions about this repository.

- **Subject:** a clean clone of commit `5c13f32`.
- **Import:** into a throw-away memory through `code_import.run` → `vi_code_sync`, in 24.6 s: 236 files, 2,814 pieces and 2,995 fact
  objects.
- **clingo:** `bridge + question` through `logic._run`, the server's path over `objects(…)`.
- **Search:** `memory.recall(query, or_fallback=True)` over the code pieces only (fact objects excluded). Top 4 is what the VI shows;
  top 10 is also given. A class piece counts for its own name.
- **Truth:** a different method from the importer's, never `vi_mcp.code_facts`:
  - line regexes for SQL and `raise`;
  - regexes over the import lines for aliases, then `alias.fn(` per line;
  - indentation for "inside `with …lock`";
  - `git diff -U0 HEAD~1 HEAD` for the newest commit;
  - Python's own import of `api_en`/`tools` for the aliases;
  - an `ast` walk only for import-time imports and body hashes.

| # | question | truth | clingo hit/answered | exact | search@4 hit/shown (P, R) | search@10 R |
|---|---|---|---|---|---|---|
| Q01 | Which functions write table objekt? | 32 | 32/32 | yes | 4/4 (1.0, 0.125) | 0.25 |
| Q02 | Which functions write table kante? | 3 | 3/3 | yes | 0/3 (0.0, 0.0) | 0.0 |
| Q03 | Which functions write table tag? | 6 | 6/6 | yes | 2/4 (0.5, 0.333) | 0.833 |
| Q04 | What calls memory.save? | 88 | 88/88 | yes | 1/4 (0.25, 0.011) | 0.045 |
| Q05 | What calls memory.new_object? | 238 | 238/238 | yes | 4/4 (1.0, 0.017) | 0.042 |
| Q06 | What calls logic.check? | 25 | 25/25 | yes | 2/4 (0.5, 0.08) | 0.32 |
| Q07 | What calls code_import.server_sync? | 1 | 1/1 | yes | 1/2 (0.5, 1.0) | 1.0 |
| Q08 | Which functions raise LogicError? | 19 | 19/19 | yes | 4/4 (1.0, 0.211) | 0.526 |
| Q09 | Which functions of vi_mcp.tools raise Rejected? | 7 | 7/7 | yes | 2/4 (0.5, 0.286) | 0.571 |
| Q10 | Which modules import vi_mcp.memory when loaded? | 120 | 120/120 | yes | 3/4 (0.75, 0.025) | 0.058 |
| Q11 | Which repository modules does clingo_object.notation import when loaded? | 0 | 0/0 | yes | no text query | — |
| Q12 | Which modules import clingo_object.notation when loaded? | 8 | 8/8 | yes | 1/2 (0.5, 0.125) | 0.25 |
| Q13 | Is there an import cycle (which modules are on one)? | 0 | 0/0 | yes | no text query | — |
| Q14 | Which tests call code_import.run? | 3 | 3/3 | yes | 3/3 (1.0, 1.0) | 1.0 |
| Q15 | Which tests call memory.save directly? | 28 | 28/28 | yes | 1/4 (0.25, 0.036) | 0.179 |
| Q16 | Which production functions write a table outside a lock? | 4 | 4/4 | yes | 1/3 (0.333, 0.25) | 0.5 |
| Q17 | Which production functions have the same body as one in another module? | 4 | 4/4 | yes | no text query | — |
| Q18 | Which functions did the newest commit change? | 6 | 6/6 | yes | no text query | — |
| Q19 | Which tool aliases point at no registered tool? | 0 | 0/0 | yes | no text query | — |
| Q20 | Which functions call memory.save while holding the lock? | 5 | 5/5 | yes | 1/4 (0.25, 0.2) | 0.2 |

**Totals.** clingo answered 20 of 20 exactly, 0.8–0.9 s per question. Search answered 1 of 15 text-answerable questions exactly, both
in its top 4 and its top 10 (Q14).

**The first run of the truth was wrong.** Its import regex lost the alias in `from vi_mcp import memory as g  # noqa: E402` because of
the trailing comment, so clingo looked wrong on Q04, Q05 and Q15 (10, 27 and 7 extra callers). After the fix (strip comments) clingo
matched exactly.

**Limits of this measurement:**
- Q11, Q13 and Q19 have an empty truth, which a program that answers nothing also matches. A non-empty cycle and a dangling alias are
  covered by the package's worlds and by `tests/test_code_facts.py`.
- The SQL truth uses the same keyword regex as the importer, only per line instead of per string constant.
- Search is measured as retrieval of the right pieces, not as an agent reading them.

## Training form (designed, not exported yet)

`code_facts.training_pairs(facts, path, object_id)` writes grounded question–answer pairs about one function straight from its facts.
No language model is involved: the facts are exact, so are the answers. Examples: where it is defined, what it calls, which tables it
writes, whether it holds the lock, what it raises. The pairs go through the one definition in `vi_mcp/qa_pairs.py`: `select` (grounded,
names its subject, numbers stated) and `split_probe` (one never-trained probe per function, chosen by the object id).

On this repository: 2,600 functions gave 5,601 training pairs and 2,335 probes, with 0 rejected. The first version lost 85 pairs to
"not grounded" because answers began with "Yes,", a word no document contains.

Not wired yet: `dataset.py` skips every `art_wissen:code` object (a copy of a repository is no knowledge). The planned step is an
opt-in stage that exports `training_pairs` of the fact objects of an index chosen by a person. The probes go to `probe.jsonl` like the
knowledge probes.

## Stated limits of the facts

Each limit is a class of miss:
- Calls are resolved through imports, module-level definitions and `self`/`cls`. A call on a local object (`o.change()`) is counted
  per function (`unaufgeloest`) and not resolved: 10,251 such calls next to 8,818 resolved ones in this repository.
- SQL is found in string constants that name the table. An f-string table name is not found.
- "Under the lock" is lexical: inside `with <x>.lock` in the same function. A caller's lock is credited by the rule (`covered`), not by
  the fact.
- `koerper` hashes the exact body. Renamed variables give another hash, and bodies under 12 AST nodes are not hashed.
- `geaendert_in` comes from `git blame` of the working tree: the commits that last touched a line of the function (at most 5, newest
  first). Uncommitted lines have none.
- Only Python files get facts. Other languages keep their text pieces.
