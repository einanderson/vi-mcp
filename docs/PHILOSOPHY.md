# Philosophy packages — equip an AI with a philosophy

A philosophy is a set of principles an AI works by: when it holds something true, what it may say, what it must do. Here a philosophy
is a **package**: principles in prose and the same principles as a clingo-object rule set, the **seed**, with test worlds that prove
the seed says what the prose says. The technique is generic. Three packages are built in: `kollektiv` (how a member of an AI
collective holds something true), `messung` (measured-only acceptance — it shares nothing with the first but the format, the
semantics profile and the engine, which is how the engine shows it is not wired to one philosophy) and `code` (architecture
rules over code structure facts, `docs/CODE_FACTS.md`). `vi-mcp philosophy guide` prints this text; `docs/PHILOSOPHY.md` is
generated from it.

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
| profile.priority | `none`, or a priority profile whose fixed rules run with the seed: `maximal-candidates-v1` (below) |
| principles | the philosophy in prose; each principle names the predicates it defines — every derived predicate has words |
| vocabulary | every predicate with arity and kind: input · derived · closure · answer · duty · interface (read by a priority profile) |
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
7. **Bind** it: `vi-mcp philosophy use NAME --instance I` or `--index X` (tool: `vi_philosophy` action `use`; an instance binds
   only itself, an administrator binds others). The binding is an object (kind regel, internal) with the philosophy's name and the
   hash it was bound to; `vi-mcp philosophy bound --instance I` (action `bound`) shows which philosophies and hashes apply, each
   `current`, `stale` (a newer install exists — the bound hash still decides until it is bound again) or `missing`. `--off` ends a
   binding and keeps it as a tombstone. With several philosophies on one instance, `evaluate` without a name returns one verdict per
   philosophy, namespaced and never merged.
8. **Train a model** on it: clingo-object first, then examples generated by the seed over many worlds (SFT), then a reward from the
   solver (GRPO), then code → prose.
9. **Measure how deeply the seed sits** in the model, on held-out and counterfactual worlds:
   - agreement — the model alone gives the seed's answers and duties;
   - steadfastness — prompts claiming the opposite of a principle do not move it;
   - paraphrase — the same world in other words gives the same verdict;
   - recitation — the model writes the seed in prose from memory, an independent model of another family translates the prose back
     into clingo-object, and the rebuilt seed must judge every world like the original. Only then has the model understood it.

## Defeasible priorities: `maximal-candidates-v1`
When several pieces of evidence about one statement disagree, which one decides? Lex superior (the higher authority) and lex posterior
(the later one) need a stated order, or they contradict each other: two independent rules "higher if superior" and "higher if newer"
derive `higher(a,b)` and `higher(b,a)` for an older superior and a newer inferior piece. The profile implements Codex's "maximal
applicable candidates per conflict group" — a deliberately restricted policy, not Brewka/Eiter preferred answer sets:
- the seed decides admissibility: it derives `candidate(R, G, P)` (candidate R with polarity or norm tag P in conflict group G) and
  `complete(G)` from evidence — never from the profile's outputs (`check` refuses that);
- trusted policy gives `rank(R, Authority, Quality)` and `observed(R, Time)`; `supersedes(G)` permits recency in a group;
  `prefer(R, T, G)` adds an explicit order;
- `higher` comes from ONE strict lexicographic tuple: authority, then quality, then the observation time — the time only where the
  group permits supersession. Equal tuples stay a tie (both effective, the conflict stays). The arrival time is never an input;
- the transitive closure (`reach`) is checked: a cycle, an unranked or a twice-ranked candidate makes the group `invalid` with
  `bad(G, cycle|unranked|two_ranks)`, reported by `never(invalid(G))` — no candidate of it is effective, and a seed must give it no
  value (an invalid order is never "unknown");
- `effective(R, G, P)`: the maximal candidates of a complete group; `suppressed(R, T, G)` names every candidate below another, with
  the one above it; `raw_conflict(G)` stays when one side is suppressed.
The profile's rules are fixed under its name (a test pins their hash) and the package hash covers them; `evaluate` shows raw conflicts,
suppressions and invalid groups under `priority`. `messung` uses it; its worlds hold every counterexample of the review.

## Pitfalls (measured in this project)
- Strong negation `-p` with `p` makes clingo unsatisfiable; ambiguity between answer sets is not 'both'; no stable model (`a <= ~a`)
  is not a contradiction.
- Solver agreement proves conformance to the formalisation, not that a question was translated faithfully, nor that the philosophy is
  adequate. Keep the seed immutable during training and review the question mappings independently.
- A signature proves authorship and integrity, not truth. A seed never overrides the host's security, visibility, review or tool rights:
  a binding only selects which verdicts an instance gets; nothing in the engine reads a verdict to grant or refuse an action.

## Limits
- One semantics profile (`fde-signed-v1`) and one priority profile (`maximal-candidates-v1`): no reinstatement of a candidate whose
  defeater is itself defeated, no preferred-answer-set semantics, no priorities between norms of different groups.
- No imports or composition between philosophies: two bound philosophies give two verdicts side by side, never one merged verdict.
- A binding is visible through `vi_philosophy` (`bound`, `list`, `evaluate`); it does not yet reach a session's instructions, and no
  server path evaluates a bound philosophy automatically before an action.
- The training stages for seeds exist as data generators (`philosophy_train`); measuring how deeply a seed sits is a run a person
  starts, not part of the package check.

