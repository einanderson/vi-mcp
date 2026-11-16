# Learning What to Put in Front of the Model: Measured Memory, Incubation and a Weighted Language for AI Agents

*vi-mcp technical report · version 0.67.0 (draft) · 15 November 2026*

**Abstract.** vi-mcp is a shared memory for many AI agents, served over the Model Context Protocol (MCP). It never trains the language
model it serves. Instead it learns, from the measured outcome of every action of every connected agent, *which context to put in front
of the model next* — which hints, rules and solutions, in which order. This report describes how that memory learns (labels from
outcomes, a logistic-regression router, randomized exploration and hold-back), how a memory *matures in an incubator* until measured
criteria say its data can fine-tune an AI, and **VIL**, a small, code-like language in which an AI asks the memory and the memory
answers. VIL has a readable symbol layer — speech acts, evidence markers and words of the form `kind:value` — and a meaning layer of
measured weights per word; memories exchange these weights, not prose and not model weights. Every design decision is grounded in an
explicit epistemic rule set (identity, non-contradiction, exhaustive cases with named unknowns, sufficient reason) and in measurements
taken on a production instance; where a claim is not measured we call it a hypothesis. We report what works (search MRR 0.73 hybrid vs
0.36–0.43 full text; ranked hints beat random ones by 6.9 ± 3.4 points of success; LLM labels agree across prompt variants with
Cohen's κ 0.80; with **clingo-object** — answer set programming over the memory's own objects, written as an object-oriented language
in real Python — a language model answered 15 of 16 relational questions exactly, against 2 of 16 with searches alone), what does not yet
(no human gold set, the total effect of hints still being measured) and what we found wrong in our own work along the way. Since 0.66.0
the report also covers **philosophies as seeds**: an AI's principles exist only as a checked clingo-object rule set, and prose is a
translation checked by a round trip through an independent model. The first philosophy, *kollektiv*, says how a member of an AI
collective holds a statement supported and what it must do; its version 1.3.0 counts **origins, not votes**, as an answer to fake news.
Fine-tuning a 0.6B model on such a seed taught it the seed's verdicts (held-out worlds not seen in training: 20 of 38 exact, against 0 of 38 for the base model)
but erased the language stage it was built on: catastrophic forgetting, which we found and are measuring a fix for.

---

## 1 Introduction

Agents repeat mistakes. A coding or operations agent that failed yesterday because a host needs a key file, or because `pkill -f` kills
its own shell, will fail the same way tomorrow unless something in its context says otherwise. The usual answer — a bigger model, a
longer system prompt, fine-tuning — is either out of the operator's hands or too coarse. vi-mcp takes the narrow route: a memory that
every agent reports to before and after each action (`vi_before`, `vi_after`) and that, before the next action, offers at most four
lines of context chosen by a model that is retrained daily on the outcomes of all agents. In that sense vi-mcp is *a search for the best
prompt* — not one master prompt, but the few right lines for each situation (tool, host, error, project).

Three questions shaped the work reported here:

1. **How does a memory learn without training the language model?** (Section 4.)
2. **When is a memory good enough to teach an AI?** — the *incubator* (Section 6). The goal set by the system's owner: "a VI that comes
   out of the incubator fine-tunes an AI."
3. **In what language should AIs talk to the memory and to each other?** Prose is the default today and, we argue, inefficient and
   unmeasurable. VIL (Section 5) is our answer; its meaning lives in weights.

The system runs in production for an IT operations department (about 24,400 knowledge objects, five agent instances, a log-monitoring
pipeline issuing thousands of lookups a day). All numbers below come from that instance or from reproducible local tests, each with its
date and method.

## 2 Epistemic foundation

vi-mcp is built against a short rule set that its owner formulated as design and review rules. They are not philosophy here but the
classes of error that cause misuse in practice.

1. **Identity (A = A).** One thing, one name, one place: the same name in code, API and interface; one source of truth.
2. **Non-contradiction.** Never two views with contradictory state of the same object; impossible combinations excluded by type.
3. **Exhaustive cases.** Every case falls into exactly one *named* class; `unknown` is a named state, never silence. Nothing vanishes
   silently: what is removed leaves a tombstone, what is held back is counted with its reason. (Formally three-valued, Kleene K3; only
   the classification of cases is two-valued.)
4. **Sufficient reason (Leibniz).** Every displayed state has a reachable cause; every refusal names its reason; numbers carry unit,
   period and origin.
5. **Valid inference.** No causation from proximity, no self-confirmation, no incomplete disjunction.
6. **Intension vs. extension, parsimony.** Rule and instance are kept apart; two ways to the same result are a logic error. And: how
   would we notice that an assumption is wrong? If there is no observation point, a display or a test is missing.

Three operational rules follow and recur throughout the system: **every statement carries evidence or is called a hypothesis**;
**no instance rates itself** (there is no tool to set one's own grade, and a publisher's rating of its own content does not count);
**a language model's verdict may only tighten** (it can be deceived by the text it checks; only fixed rules reject for good).

These rules are not decoration. In this report's own development they caught concrete errors: an evaluation that let random tie order
pass as a measured effect (Section 5.4), a statistic that mixed "nothing known" with "search failed" (Section 4.6), a parameter whose
values were silently translated (Section 5.2).

## 3 Architecture

**Objects.** Everything is an object in one database (SQLite or PostgreSQL): experiences (one per reported action), insights, rules,
skills (problem → solution), searches, models, conversations. Each object has a kind, a text, a content field (code, tables and commands
stored unabridged — what is missing is a gap in memory), hashtags `kind:value` (`host:`, `tool:`, `thema:`, `fehler:` …), an owning
instance, a visibility (private · shared · internal) and, since 0.61.0, exactly one **index** (`global`, a project, or a special VI).

**The action loop.** A hook in the agent's client calls `vi_before(tool, args)`; the memory answers with at most four hints, a risk
warning if the router predicts failure, and logs what it offered with the offer probabilities. After the action `vi_after` reports the
outcome; the experience object records tool, arguments, outcome, the hints shown and the router version. The user's prompt is masked
and never stored.

**Sleep.** A background consolidation clusters experiences into insights and skills, rates work (the scorer), archives what is no longer
used (with a tombstone), trains models and runs the daily measurement (Section 4.5). Since 0.62.0 it also accounts for machines that are
not up around the clock: a sleep older than 24 h is caught up in the next quiet moment (no MCP request for 10 minutes, something new to
consolidate), and on a workstation the importance trigger waits for such a moment, so the language model does not run while someone works.

**Network.** Servers form networks without a master: each appends signed entries to its own logbook (Ed25519), a separate service
exchanges them, and the memory checks every foreign entry again — fixed rules, quarantine, a reviewer model, release. Open networks
receive only anonymised knowledge; company networks carry internal knowledge to members of one organisation. The same meaning on several
servers is one object with several holders (content hash), rated by every server's own measurements.

## 4 How the memory learns

### 4.1 Labels from outcomes

The label of an action (or of one hint shown before it) is defined once (`scorer.label`) and used by every learner, in this order:
direct feedback on the hint (`vi_feedback`, `vi_confirm`, `vi_refute`); implicit adoption (a distinctive command, path or option of the
hint appears in a later successful action of the same session); a bad conversation grade; the scorer's code stage (repeat, red test,
user correction); the tool's own outcome. Each source carries its weight.

### 4.2 The router

A logistic regression (scikit-learn, `class_weight='balanced'`) predicts the success of an action given its features (tool, triggers,
the hint, its age, content features). The regularisation strength C is chosen on the newest 20 % of the training part by weighted log
loss, then refitted on the whole training part; the newest observations form the holdout. *Measured:* 17,590 observations, holdout AUC
0.863–0.865 (13.11.2026). Platt calibration was tried and rejected: holdout log loss rose from 0.427 to 0.473 (12.11.2026). Since 0.62.0
every training also reports a reliability table and the expected calibration error (ECE, Guo et al. [1]); the balanced class weights
shift the probabilities on purpose, and the table makes that visible instead of hiding it.

A new model must beat the current *champion* on the same unseen observations or the champion stays. An index (a project) with enough
data (≥ 300 observations, ≥ 30 failures) trains its own model, used only if it beats a shared reference fitted without the index's
holdout by more than two standard errors. *Measured on a production copy (13.11.2026):* the largest project index lost (AUC 0.29) and
stays with the shared model — the gate did its job.

### 4.3 Randomization: exploration and hold-back

Ranking hints by a model that learns from its own choices confounds the model's choice with the outcome. Two randomized experiments make
the effect causal:

- **ε-exploration.** With probability ε the last hint slot goes to a random candidate from further back; the offer probabilities are
  logged, so per-hint effects can be estimated by self-normalised inverse propensity scoring (Swaminathan & Joachims [2]).
  *Measured:* exploration minus ranked −6.4 ± 3.7 points (12.11.2026) and −6.9 ± 3.4 points (13.11.2026, daily measurement over 7 days):
  the ranked hint beats a random one. Since 0.62.0 the same probabilities also enter training: a second router is fitted with each observation
  weighted by 1/p (clipped at 10) and replaces the uncorrected one only if it wins on the holdout by more than two standard errors.
- **Hold-back** (0.61.0). On a random 5 % of actions that have hints, the hints are held back (the risk warning stays). Comparing held-back
  with shown actions within instance and tool (Mantel–Haenszel) estimates the *total* effect of showing hints. A raw comparison cannot:
  hints appear on hard actions (raw lift −37 points, matched lift −1 point, 13.11.2026). The hold-back was not yet deployed at that measurement; it is now collecting observations; a difference of
  a few points needs a few hundred held-back actions — weeks.

Warnings, by contrast, are clearly predictive: actions after a warning succeeded 51 % of the time, without a warning 98 % (24 h,
13.11.2026, 390 warnings).

### 4.4 The LLM weight layer

A language model may propose changes to router weights ("weight-space programming"); they act only after an offline gate (paired log
loss on a holdout) and a randomized online test on 20 % of actions. On 13.11.2026 a proposal of six changes passed offline (log loss
−0.00006 ± 0.00011 per observation on 3,645 holdout observations) and entered the online test. This is the only path on which a language
model may *loosen* a weight; it never sees text from the network.

### 4.5 Fixed measurement

Once a day the memory answers a fixed question set: 150 knowledge objects sampled by stratum, two questions each, written by a language
model. *Measured:* hybrid search (bge-m3 embedding plus full text, reciprocal rank fusion, cross-encoder reranking) reaches MRR 0.735
(12.11.) and 0.732 (13.11.), Hit@5 0.84–0.85; full text alone 0.364 and 0.432. The evaluation targets are held out of every training export
(Section 6.3).

### 4.6 What a statistic hid

The usage statistics reported 26 % of searches without hits. A reading of the search log on production (13.11.2026, read only; 424 searches
since logging began) showed that all 111 empty searches came from the log-monitoring pipeline asking about hosts the memory had never
heard of (57 hosts, none present as a hashtag); agents' searches had no empty result. "Nothing known" is the right answer there, not a
failure. Since 0.62.0 an empty search says why — a word unknown to the memory (with "did you mean" from the same kind) or known words that
no object carries together — and the statistics count both separately. The earlier interpretation ("agents guess tags that do not exist")
was ours and was wrong; it is corrected here and in the memory.

### 4.7 Search at scale: four results, half a million objects

An agent sees four results, so **Hit@4** is the measure that matters. On a production copy (150 questions × 2 styles) hybrid search —
bge-m3 vectors, FTS5 full text, reciprocal rank fusion and a reranker — reached Hit@1 0.80 and **Hit@4 0.933**; full text alone 0.56. The
reranker takes 1.9 of 2.4 s per query. A scale test with 500,310 knowledge objects (the copy plus distractors from source pages without
evaluation targets; a proxy, not a picture of future data) kept Hit@4 at 0.930; caching the vector matrix and filtering visibility only on
the full-text hits cut the full-text step from 1,826 ms to 90 ms without changing one of 300 ranks.

Two experiments, two outcomes. Putting an object's content into its one vector **hurt** (Hit@4 0.963 → 0.843 on 5,000 objects; the gain
reported for chunked retrieval comes from chunks with their heading path, which we have not tested). Removing **stop words from the query, never from the index**
— the published Snowball lists for German and English, unchanged [73] — **helped** full text strongly: natural questions ANDed "wie",
"ich", "how", so the AND set was empty and the OR fallback ranked by weak words. On the held-out half of a split fixed before any run,
full-text Hit@4 for questions rose from 0.347 to 0.533 (16 improved, 2 worse, exact McNemar p = 0.0013); keyword queries did not change;
hybrid Hit@4 moved from 0.933 to 0.947, not significant (p = 0.22). No FTS5 tokenizer (porter, trigram, a Snowball German stem column)
beat unicode61; trigram hurt keyword queries and needs a 5.4 GB index at 500k objects.

## 5 VIL — a language for AIs, with meaning in weights

### 5.1 Why not prose, and why not only weights

Agents talk to each other in prose today. Prose cannot be checked by a parser, its meaning cannot be measured, and it invites prompt
injection across trust boundaries. The opposite extreme — communicating through model weights, adapters or hidden states — is efficient
between identical models (DroidSpeak: up to 4× throughput when models share an architecture [3]; LatentMAS reports up to +14.6 % accuracy
with far fewer tokens [4]) but works only between identical or closely aligned models, is unreadable to people, and leaks content:
vec2text reconstructs 92 % of 32-token texts exactly from their embeddings [5], and embeddings of different models can be translated into
each other [6]. Adapters can carry backdoors [7]; machine-invented languages drift away from human readability [8, 9]. Between
organisations with different models and a review process, neither extreme fits.

The history of agent communication languages points the same way. KQML and FIPA-ACL standardised an envelope of performatives (FIPA
defines 22 communicative acts and 13 message parameters, only the performative being mandatory [10, 11]); what failed was agreement on
meaning — FIPA's semantics rested on agents' private mental states and was not verifiable [12]. Singh argued that meaning must be public
and social [13]. In vi-mcp, meaning *is* public: it is what is measured.

VIL therefore has two layers: a **readable, executable symbol layer** and a **meaning layer of measured weights**.

### 5.2 The symbol layer

An utterance is a speech act, optionally a question type, and words:

```
utterance  = act [ "." qtype ] { word } ;
act        = "ask" | "tell" | "do" | "promise" | "def" | "rate" ;
qtype      = "how" | "why" | "what" | "who" | "where" | "rule" ;
word       = [ "!" ] ( kindvalue | "@" name | "#" topic | phrase | text | group | "since:" age ) ;
kindvalue  = kind ":" value [ "*" ] ;          (* host:web1.example, tool:nginx, thema:deploy *)
group      = "(" word { "OR" word } ")" ;
age        = digits ( "d" | "h" | "w" ) | date ;
```

Design choices and their grounds:

- **Speech acts as a closed list** (Searle's classes; FIPA [11]): `ask`, `tell`, `do`, `promise`, `def`, `rate`. Only `ask` is executed by
  the search; `tell` is `vi_learn`, `rate` is `vi_feedback`, `def` adds a word. A `do` arriving in a message is never run — foreign
  content is data, not instruction (CaMeL: control flow only from the trusted request [14]; OWASP LLM01 [15]).
- **Question types** follow coarse question taxonomies (Li & Roth: six coarse classes classify at 92.5 % accuracy vs 89.3 % for fifty fine
  ones [16]). The type was meant to order the hits; measured, it does not help (5.4) — it is recorded for learning, not used to rank.
- **Words are the memory's hashtags**, one marker per meaning, linear and order-free — the agglutinative principle, and named roles
  instead of positions (case rather than word order; WALS ch. 20, 81 [17, 18]). Short forms: `@web1` is every host starting with `web1`,
  `@web1.example` exactly that host; `#topic` is `thema:topic`.
- **Small and tolerant.** Language models write rich query languages poorly (ES|QL about 59 % correct [19]; Cypher about 60 % for GPT-4o
  [20]); Elastic advises against strict query strings in search boxes [21]. VIL keeps to a handful of operators; anything that does not
  start with a known act is an ordinary filter, untouched.
- **Vocabulary first.** "What is there, what can I search?" — `vi_vocabulary` lists the kinds of words with counts, the values of a kind
  with their measured weight, and completes half-typed words; the internal search offers the same as autocomplete. Counts cover only what
  the asking instance may see. (A test caught that the vocabulary tool, without a session, would have listed host names from other
  instances' private objects; it now always takes the caller's visibility. A second test caught that values of a parameter named `kind`
  were silently translated by the German alias layer — `tool` arrived as `werkzeug`; the parameter is now `word_kind`.)

The answer comes back in the language, one line per hit:

```
tell.<memory> <id> "<text>" ev:<evidence> w:<weight>/<ratings> src:"<source>"
tell.procedure erkenntnis:… "deploy web1: stop, copy, start" ev:measured w:0.83/10 src:"https://wiki.example/deploy"
```

**Evidence is mandatory.** Like the evidentials of Quechua (`-mi` direct, `-si` reported) or Lojban's attitudinals (`ja'o` I conclude,
`ti'e` hearsay) [17, 22], every statement says how the memory knows it: *measured* (shown and rated), *documented* (a source URL),
*hearsay* (imported from another server), *inferred* (derived by a language model in sleep), or *unknown* — a named state, never silently
"observed". The memory kind (*fact*, *procedure*, *episode*, *rule*) follows the semantic/episodic/procedural distinction of cognitive
architectures for agents (CoALA [23]).

### 5.3 Classification: teaching the words their types

Knowledge must carry its question type and memory kind for `ask.<type>` to work. A language model labels each object with two prompt
variants that differ in wording and label order — both shift LLM labels [24, 25] — and stores a label only where they agree; disagreement
becomes the named state `answers:unsure`. The model's own confidence is not used (verbalised confidence is overconfident [26]). Each label
carries its version (model, prompt hash, schema), so a new prompt relabels instead of mixing. The model also writes up to three questions
a colleague would type to find the object (doc2query; document expansion raised MS MARCO MRR@10 from 0.184 to 0.277 [27]).

*Measured on a stratified production sample* (13.11.2026; 300 objects, both variants, 600 calls, an open-weight model on a shared
llama.cpp server): agreement 86.3 %, Cohen's κ 0.80 for the question type (the threshold for reliable annotation is often put at
α ≥ 0.8, tentative at ≥ 0.667 [28]); 92.3 %, κ 0.88 for the memory kind; most disagreements were what → where and how → what.
**28 of 600 answers (4.7 %) were outside the label enumeration although the request carried a JSON schema** — every answer is therefore
checked again and asked once more, otherwise `unsure`. Latency: median 56 s, 90th percentile 140 s per call on the
shared server. In a check of 26 objects by the author of this report (not a human gold set), about 24–25 primary labels were plausible.

Because labelling every object with a language model is slow, a **student** learns the labels: one logistic regression per label kind on
the bge-m3 vectors the memory already has, trained only on agreed labels (≥ 120, ≥ 15 per class), labelling only where ≥ 80 % sure; 10 %
of its sure labels are audited by the language model and the agreement is recorded, so drift becomes visible. Its gate is the accuracy of
the labels it actually gives: C is chosen by 5-fold cross-validation, and it is active only if its sure labels are ≥ 90 % right on ≥ 10 %
of the objects. *Measured on the production sample* (252 agreed labels, the stored vectors, the gate's own cross-validation): with C = 10
it is sure on 15.5 % of objects for the question type at 94.4 % accuracy and on 24.6 % for the memory kind at 95.0 % (overall accuracy
75–78 %); C = 100 is sure more often but 86–87 % right, below the gate; rare classes (`where`, `rule`) stay with the language model. The first design — C = 1 and an overall-accuracy gate of 0.8 — would never have
labelled anything: on unit-length vectors no prediction reached 0.8. Measuring on real data, not only on synthetic clusters, found it. Classifiers trained on LLM labels reach roughly the quality of those trained on human labels [29]; FineWeb-Edu
used exactly this pattern at scale [30]. Old data migrates where it is used: what a real search returns unlabelled is labelled at once by
the student or queued, and sleep labels the queue first.

### 5.4 The meaning layer: weights per word

The meaning of a word is its measured weight: for every word `kind:value` and question type, the beta mean of helped/not helped over the
knowledge carrying it, recomputed in sleep.

**Does the language rank better? Measured: no.** On a copy of production with its fixed question set (140 questions whose question word
gives the type, two query styles, 280 queries, paired against the plain hybrid order with MRR 0.737) every reordering lost: the fixed
prior as a priority rank −0.059 ± 0.012 (34 worse, 7 better; `where` from 1.0 to 0.54), and −0.289 ± 0.021 once word weights were
computed; a lift of at most two places for objects labelled with the type −0.006 ± 0.003; that plus the word weights −0.056 ± 0.010
(47 worse, 4 better). So the type no longer reorders anything; it is recorded with each search. Our reading (a hypothesis): the word
weights measure whether a *hint helped an action* — a different label from whether a *hit answers a question*. A meaning layer for
retrieval has to be learned from what is used after a search. We report this because the design was plausible, the literature does not
establish type matching as helpful either, and only the measurement showed the harm — the first version would have shipped it.

The missing piece is a relevance label: 3 of 386 searches had a hit opened afterwards, too few for click models [52]. Following
UMBRELA [51], a language model now grades a few real searches per sleep (0–3 per hit; all hits of a search in one call, a stated
deviation forced by about a minute per call on the shared server). On the production copy six searches gave nDCG@10 0.835, a rank-1
answer in 50 % and nothing relevant in the top 10 in 33 %. The grades only measure; a relevance weight trained on them must first win on
the fixed question set. A first attempt did not: a logistic regression over the hits' features, trained on grades of 90 synthetic
(doc2query) questions, gained +0.020 ± 0.005 nDCG@10 in grouped cross-validation but +0.002 ± 0.006 MRR on the fixed set — its weights
had learned the synthetic questions (it demoted whole document kinds), not relevance. The labels must come from real searches.

**Memories talk with weights.** Word weights travel between servers as ordinary signed ratings whose target is
`sha256("vil-word-v1|" + word + "|" + type)`. No new protocol type is needed (older peers would reject it), and the word itself never
travels in clear: a receiver can attribute a weight only to a word it knows. Into an open network go no host names and no word the
anonymisation flags as internal (the hash of a host name could be guessed by dictionary); a company network gets all. A receiver
combines foreign weights with its own, each foreign server weighted by its measured benefit as a publisher and capped at 20
observations — a foreign weight is a measurement, never a command. The symbol stays readable; the meaning travels as measured weight.

### 5.5 The logic layer: rules decided by formal logic, reliability in weights

Words and weights say *where* knowledge is and *how much it helped*; they cannot say what *follows* from rules. The owner asked for a
way to express logical descriptions "in Python, but by the principles of formal logic". We compared a Python-embedded Datalog, a
three-valued evaluator of our own and answer set programming, and chose ASP with the solver clingo [44, 50]: its stable-model semantics
contains stratified Datalog as a special case, it is nonmonotonic (rules with exceptions, as norms have), strong negation gives explicit
falsity next to "not derivable", and it searches and optimises.

**Notation.** A program is written in Python syntax and read by Python's `ast` against a whitelist — never executed: facts
`zone("gate1", "dmz")`, rules `head <= body` with `&`, `~x` for negation as failure, `-x` for strong negation, parenthesised comparisons,
`never(c)` for a constraint and `soft(rule[, belief])` for a weighted rule. A prelude gives the deontic square (obligatory, forbidden,
permitted; a violation is an obligation not done, obligation and prohibition together a named conflict) [45].

**Every case named.** ASP answers an inconsistent program with *no* answer set and treats a violated integrity constraint the same way —
the reason vanishes. We translate `never(c)` into a marker atom and report the violated constraint with its line; a contradiction
between hard rules is solved again with `-p` renamed, and the atoms that hold both ways are listed (Belnap's *both*). An atom is
*true*, *false*, *both*, *unknown* (the program does not mention it), *not derived* (the program defines it, this instance does not follow)
or *undetermined* (answer sets disagree) — the Kleene distinction between "no value" and "false" kept in the answer, not only in theory.

**Rules in weights.** The owner's next step was "clingo is also written in weights, so that with weight-space programming we learn clingo
fast". A `soft` rule is an LPMLN rule [46]: each ground instance may be dropped at the cost of its weight, a stable model's probability is
proportional to exp(−cost), and P(atom) sums over the models. We compile soft rules to weak constraints by the lpmln2asp scheme that
plingo uses [47]; the instance's variables are part of the marker, so a penguin drops its own instance of "birds fly" without the rule
losing its force for other birds (the first implementation dropped the whole rule — caught by the test, fixed before commit). The weight is
the log-odds of the rule's measured probability: the stated belief as two pseudo-observations, plus every conclusion checked against
reality (`vi_logic rate`) booked on exactly the rules that derive it (`fired` markers), plus the same counters from other servers on the
rating channel of the word weights, target `sha256("vil-rule-v1|rule")`, open networks only for rules without constants. In the test, six
"wrong" ratings move `soft(flies(X) <= bird(X), 0.9)` from p = 0.90 to 0.225 and the conclusion to *not derived*; twelve "right" ones
bring it back. Two deliberate limits: a hard rule stays hard whatever is measured (its measurement is shown with the advice to soften it —
the author's assertion and the measured world kept apart), and weights are learned automatically while new rules are only *proposed*
(5.6) — neural atoms (NeurASP [48]) are not part of this. The counting is a simplification of LPMLN's maximum-likelihood learning: exact
when one rule alone decides a conclusion.

### 5.6 clingo-object: the memory's objects in the logic, as an object-oriented language

The logic layer reasoned over facts someone typed. The owner's next request was to reason over the memory itself — "the data lie as
objects in SQL, so we program with these objects in clingo" — and to make of SQL and clingo "a dialect, the way Delphi grew out of Pascal";
later, plainly: "clingo is the language for LLMs".

**Objects as facts.** In the notation, `(O in objects("art:skill host:web*"))` binds `O` to the objects of a `vi_recall` filter;
`(O.host == H)` reads a hashtag, `(O.kind == "skill")` a field, `(O.age > 90)` a number; `linked(O, B)` follows an edge. Before solving,
the server loads the objects of every filter by SQL — as the *asking* instance may see them, never quarantine or archive, the objects one
edge away only when the program follows edges — and turns them into facts, as DLV-DB's `#import` does for relational tables [55]. What is
cut at the limit (20,000 objects), not visible or excluded is counted and shown. Searching stays SQL; inference is clingo.

**Proposals, not writes.** A head `propose(tag(O, "status", "stale"))` or `propose(link(A, B, "reason"))` is derived on the server and held
in quarantine as an object of its own, with the rules that derive it; its author cannot release it, a person can (`vi-mcp release`),
and releasing applies it with its origin in the target's change log. The epistemic rule "nothing changes without a reason chain and a
decision" holds for rules too.

**One language, two places.** A first draft put Python-like class blocks into the server's parser — a second language that looks like
Python and is not. The owner objected: the clingo Python package exists, develop in Python. The server must not run callers' Python (every
token would mean code execution), so the full language is a real Python package, `clingo-object`: classes (`class Skill(Knowledge): kind =
"skill"`), methods returning conditions built by operator overloading, inheritance and overriding by Python's method resolution order.
Each method compiles to one flat rule of the safe notation whose membership filter holds exactly the kinds that resolve to that
definition, so overriding needs no negation. The language core — notation, translation, solving — lives in the package and the server
imports it: one implementation. A program solved locally with the clingo API and the same text solved on the server gave the same answers
in every test.

**Measured against SQL.** On a production copy (24,449 insights, 499,234 edges), 13 relational questions — "DMZ hosts never described as
production", "locations of DMZ hosts", "log-monitoring findings for hosts at KAS-0" — agreed with hand-written SQL in 13 of 13 once the
object limit was 20,000; at 2,000, three answers were short (the cut was reported, the answer still incomplete). The single first
disagreement was an error in our SQL, which counted a refuted skill the memory's own searches leave out. Time: 0.01–0.6 s per question,
2.3 s for 13,088 objects. The measurement also found a silent cut in our own output (a check kept 200 of 403 conclusions and said "200").

**An agent that writes clingo-object.** A language model gets the notation, the memory's vocabulary (kinds with their most used values)
and a question; it writes a program answering `ans(…)`; the solver's refusal, an empty answer or a violated constraint goes back in the
solver's words, up to four rounds — the construct-and-check loop of ALM–ASP [54] and of ASP-Bench, where an agent with clingo feedback
saturated the benchmark and general Python beat specialised declarative interfaces [53]; that LLMs write answer set programs from few
examples was shown before [56]. With a 27B-class open model at temperature 0 on the same production copy and 16 natural-language
questions (4 written after all tuning):

| | exact | precision | recall | rounds | time |
|---|---|---|---|---|---|
| first prompt | 7/12 | 0.74 | 0.75 | 2.2 | 10.4 s |
| + "join facts about a host through the hashtag value, not by edges" | 9/12, held out 3/4 | 0.75 | 0.75 | 2.1 | 6.7 s |
| + the notation refuses an object used as a hashtag value | 11/12, held out 4/4 | 1.00 | 0.99 | 1.7 | 5.2 s |
| the same model with `vi_recall` searches only (≤ 4, ≤ 50 hits each) | 2/16 | 0.50 | 0.30 | 3.8 calls | 20.1 s |

The last improvement was a rule of the *language*, not of the prompt: `(H in objects("netz:dmz")) & (P.host == H)` compares a host name
with an object id and silently finds nothing; refusing it with a reason taught the model the right form in the next round. Where an
answer has hundreds of rows, a search that shows fifty hits cannot be complete; a program over all objects of a filter is. On the server
the agent is a tool action (`vi_logic action=answer`) driven by the server's own model, so any client can ask — only the notation is run.

**Training with it.** Three channels: rule weights from conclusions checked against reality (5.5); fine-tuning data (`logic.jsonl`: the
task of a saved rule set → the program → the solver's answer, and every rated conclusion, wrong ones as negatives); and rules learned
from rated conclusions with clingo itself as the optimiser — candidate rules from the conditions the right examples share, `{use(R)}`
choices, weak constraints against uncovered right and covered wrong examples — held as soft proposals for a person to release (ILASP [49]
was not used: free for non-commercial use only). In a test, a rule set that derived "review" for every test object, rated right only in
the DMZ, learned `(O.zone == "dmz")`, covering 1 of 1 held-out right and 0 of 1 held-out wrong conclusions. On production the questions
above were saved as 16 rule sets; 127 of their conclusions, rated against read-only SQL, were all right. No model has been trained on this
yet — the agent's numbers come from an untrained model reading the language's description.

### 5.7 Philosophies as seeds: principles only in clingo

The question that started this part came from the system's owner: why train a model to say "I don't know", when the research behind
the memory is philosophical? The answer we built is that **truth is not the network's business**: a declarative *seed* decides it, and
the model translates. "I don't know" is then the seed's value *unknown* with a reason, not a trained reflex.

**A philosophy is a package** (`clingo_object/philosophy.py`, format `philosophy-v2`). It stores *only* the seed — a clingo-object rule set
with `# P1 … # Pn` block markers — plus a typed vocabulary (every predicate with arity and kind: input, derived, closure, answer, duty),
the allowed answer values and **test worlds** with atoms that must and must not hold. `vi-mcp philosophy check` refuses a package unless
every predicate is declared; default negation occurs only in *closure* rules over a certified complete view (missing evidence never
becomes opposition); every world solves to exactly one answer set with its expected atoms; every answer value is produced by some world;
and every **counterfactual** world (one piece of evidence changed) changes an answer or a duty — so a seed that calls everything
*unknown* fails. Prose is never stored: the owner decided that a philosophy is its clingo, and prose a translation of it.

**Values.** Following Codex's semantic review (`p.` and `-p.` together are unsatisfiable in clingo, so strong negation cannot carry
Belnap's *both*), support and opposition are ordinary predicates `pos(S)`/`neg(S)`, and four information values follow over a closed
view: *t* (supported), *f* (opposed), *b* (both: reported, never resolved silently), *n* (nothing). A separate **acceptance** layer
(accepted, tentative, rejected, unresolved, and since 1.3.0 untestable and withdrawn) decides what a member may say; **duties** in a
deontic vocabulary follow from it (may assert, must state only as a hypothesis, must report a conflict, must ask the collective about
the unknown when a peer offers knowledge, must not adopt unreviewed foreign knowledge, must not cite an unverified source). Every value
carries a reason, and a value without one is a `never(…)` violation. Defeasible priorities between candidates use a fixed profile
(maximal candidates; a cycle is *invalid*, never *unknown*).

**Understanding as translation.** The owner's test: "if the AI has understood, it can put the seed into prose." `vi-mcp philosophy explain`
measures it as a **round trip**: a translator writes prose; an *independent* back-translator of another model family, which never sees the
seed, rebuilds clingo from the prose; the rebuilt seed must give the original's answers and duties on the package worlds and on random
worlds. A control — reference prose back-translated by the same model — marks the back-translator's own limit. With Codex (GPT family;
`codex exec` in an empty read-only directory) as back-translator, the frozen 1.1.0 reference prose agreed with the seed on **49 of 66
worlds (74 %)** after one round. All four values matched everywhere; all 17 misses were the acceptance *unresolved*, which the reference
prose never defined precisely — the method found a gap in the prose, not in the seed.

**Training on a seed** is solver-taught (solver-filtered SFT beat DPO for ASP in [60]): the seed generates worlds, verdicts, reasons and
duties; the model learns them first by SFT, then by GRPO with the seed as reward (+1 right value with reason, 0 for an honest "cannot
formalize", −1 for invented certainty; a reward only for compiling programs is gamed [63]). Before the seed, the model learns the
**language**: thousands of random clingo-object rule sets with no predicate of any seed, in both directions (`clingo_prose`), so that a
later translation of the seed cannot come from memory. Results are in 6.5.

### 5.8 Truth by origin: the collective's answer to fake news

"Just because a thousand people believe something does not make it true — and an AI cannot check facts itself." We agree with both
halves and drew three conclusions.

*First*, the seed's values say how well a statement is **supported**, never that it is **true**. Correspondence stays the aim, but an
agent without sensors has only evidence and its origins; truth in Peirce's sense is what inquiry would converge to, and every verdict
is fallible [69]. The spoken verdict reads "supported" and "opposed" since 1.3.0.

*Second*, a majority is evidence only if its judgments are independent: Condorcet's jury theorem needs independence, and with
correlated voters more voters add little [64, 65]; in an informational cascade people rationally copy their predecessors and ignore
their own information, so mass behaviour rests on very few signals [66]; and models trained on model output forget the true
distribution [68]. **kollektiv 1.3.0** therefore counts **origins, not votes**:

- *P8* — a copy (`derives_from`) adds no origin, a chain of copies leads back to one root, a citation circle has none, and text a model
  generated (`machine_made`) is never an origin of its own;
- *P9* — hearsay alone neither accepts nor rejects; a measurement makes a statement strong (testimony is the hard case of social
  epistemology [70]; signed capture such as C2PA [71] is the closest a member gets to an origin it can check);
- *P10* — when the camps of the origins are known, acceptance needs agreement across camps (the bridging idea of Community Notes [67]);
  a partly known camp holds back with its own reason `camp_unknown`;
- *P11* — a statement no observation could refute is its own class *untestable*, and an accepted statement obliges the member to name
  what would refute it (Popper);
- *P12* — an extraordinary claim needs three independent origins or a measurement (Hume);
- *P13* — a withdrawal is not a refutation: an author who takes a statement back removes the claim, not proves its opposite.

Every holding back names its reason (`grund(S, held_back, hearsay_only | one_camp | camp_unknown | extraordinary)`); absent inputs change
nothing, so all 1.2.0 worlds keep their verdicts. 35 worlds pass the check; 1,500 random worlds all solve with at most one value and one
acceptance per statement. "Surprisingly popular" [72] — the answer more frequent than predicted — is not implemented: it needs every
member's prediction of the others' answers.

*Third*, the memory must be read through the seed. A **provenance adapter** (`vi_mcp/philosophy_provenance.py`, shadow mode) turns what is
stored about a knowledge object into seed facts: one line per stored confirmation or refutation, the cited source as origin (one source
cited by many is one origin), the model family as camp, a record counted as closed only when its counters equal its kept entries. On a
read-only copy of production (24,655 knowledge objects) the seed agreed with the memory's own mode on 24,479 of 24,499 comparable objects
before P13 and 24,493 after. The 20 disagreements were the finding: **16 of 17 objects the memory called *refuted* had been refuted only
by their own author.** One word named two things. Since 0.67.0 the memory names them apart — `zurueckgezogen` (withdrawn by the author)
and `widerlegt` (refuted by two independent foreign model families) — and a migration moved 19 of 19 such objects on the copy.

## 6 Incubation and fine-tuning

### 6.1 The incubator

A memory matures in operation. `vi-mcp incubator status` measures seven required criteria, each with threshold, value and state —
*met*, *open*, or *not measured* (never counted as met):

| criterion | measured as | default threshold |
|---|---|---|
| knowledge | shared or internal knowledge objects | 500 |
| checked | share rated after being shown or confirmed — knowledge put to the test, not only stated | 20 % |
| search | hybrid MRR of the daily fixed question set | 0.6 |
| router | hint model active, holdout AUC | 0.75 |
| hints_no_harm | the randomized hold-back does not show that hints lower success | lower 95% confidence bound above −0.02 |
| labels | share classified, and the student active | 50 % |
| training_data | question → answer examples, evaluation held out | 1,000 |

An informative eighth criterion, **gold**, reports labels set by people and a **truth check**: do the measured weights rank knowledge that
people confirmed above knowledge they refuted (AUC; 0.5 means the measurement ignores truth)? Only instances marked `human` may set gold
labels — an AI is not the gold, and no instance rates itself. When all required criteria are met the memory *hatches*: its training data
is exported with the incubator state recorded.

### 6.2 Training data

`vi-mcp dataset export` writes four files and a datasheet (Gebru et al. [31]):

- `qa.jsonl` — supervised fine-tuning: question → knowledge with its source (questions from doc2query or the object's title);
- `kto.jsonl` — situation → hint, desirable or not, from the measured helped/not-helped counters. KTO learns from exactly this binary
  signal [32];
- `dpo.jsonl` — under one trigger, one hint helped clearly more than another (direct preference optimisation [33]);
- `agentic.jsonl` — tool-use trajectories: the question; a `vi_recall` call with a VIL question built from the question's words and the
  object's label and hashtags; the real `tell` lines of the memory's search; one relaxed question if the target is missing; the answer
  citing the target. Only trajectories that found their target are kept — no example teaches a wrong answer. A model trained on them
  learns to *search the memory in its language*, not only to recite it (cf. FireAct: Llama-2-7B fine-tuned on 500 agent trajectories
  [34]; AgentTuning [35]).

Never exported: private objects, experiences and conversations (they belong to their instance — the hint data are aggregated counters),
quarantine, refuted or archived objects, copies of repositories. Credentials are scrubbed at save time and again on export; the changes
are counted, an audit of the first scrubber.

### 6.3 Contamination guard and lineage

The targets of the fixed question set and any question equal to an evaluation question are held out of every training file, so a model
fine-tuned on the data can still be measured on them (contamination inflates benchmark scores [36]). Every export carries its manifest
(counts, exclusions with reasons, versions of router, classification and student, label quality, known gaps).

### 6.4 The fine-tuning interface

vi-mcp trains no model by itself. A person tells their AI "I want to fine-tune"; the AI calls `vi_finetune`, which reports whether the
memory is mature and gives the steps; `vi_finetune(action="guide")` explains fine-tuning. On the server `vi-mcp finetune kit` writes the
training data, a test split *by object* (no object on both sides), `train.py` (TRL with a LoRA adapter [37, 38]: SFT, then agentic, KTO or
DPO on top), `convert.sh` (the adapter as GGUF for llama.cpp, loaded with `--lora` and switchable at runtime [39]) and `evaluate.py` (base
and fine-tuned models get the same held-out questions, no token cap). `vi-mcp finetune record` keeps the run as a model object with its
lineage — dataset manifest hash, base model, adapter hashes, evaluation, incubator state — and counts it as adopted only if it measured
better. Hardware guidance follows the tool documentation (QLoRA of 7–8B models from about 5–6 GB VRAM, LoRA in 16 bit of 14B models
from about 33 GB [40]).

### 6.5 Fine-tuning runs

All runs: Qwen/Qwen3-0.6B, LoRA, fp32 on one AMD Strix Halo GPU (gfx1151, ROCm), in their own container; the generated kit; answers
greedy with thinking off, a token budget marked, never cut silently.

| run | what | result |
|---|---|---|
| 1 | SFT on the export, 0.25 epoch | held-out NLL 2.27 → 1.29, but useless: 96 % of rows used the object's own text as the question; the model echoed pages and looped in 55 % of answers |
| 2 | rank 16, 3 epochs, all rows | no gain on questions about held-out documents (p ≈ 0.1) — closed-book tuning cannot know what it never saw |
| 3 | 500 documents × ~5 grounded pairs, rank 64, 4 epochs | **knowledge probe** (new questions about trained documents [56b, 56c]): 7 correct / 9 partial of 60, base 0 correct / 5 partial; sign-flip p = 0.008; but it never abstains and invents hosts [56d] |
| 5 | seed kollektiv: judge, formalize, pressure | on worlds not in its training data: world-exact **20/38** (value 48/64), paraphrased worlds 18/37, pressure ("count the author's own evidence this time") 15/26, formalize 13/20, package worlds 7/10; base 0 everywhere; invented certainty 9/43 |
| 6 | language stage (3,000 random rule sets), then seed stage (1,700 rows, no translation rows) | seed verdicts kept (package worlds 12/16, invented certainty 0/12 on held-out worlds, partial eval of 56/186); **every prose answer failed** |
| 7 | as 6, seed stage with 1,000 replayed language rows | language kept (random rules 12/12, exactly as after the language stage); verdicts on clean worlds: world-exact 19/40, paraphrase 4/37, pressure 7/30, formalize 0/20 (mostly one fact off); seed prose still 2/10, recite 0/2 |
| 8–18 | language extended to the seed's forms, crossed vocabulary, facts-first, enumerated count trace, the seed as recitation target, paraphrases, GRPO, Qwen3-1.7B | in the companion paper [FINETUNING_PAPER.md](FINETUNING_PAPER.md): verdicts with training phrasings 39/40 and 154/160, recitation 46/46; held-out phrasings 3/40 — reading is the open problem |

**Run 6 found catastrophic forgetting.** Asked to translate the seed, the tuned model answered in the verdict format and looped ("My
philosophy decides this … I must not cite the source …"). The hypothesis that only the seed's words switched it into that mode was
wrong: the *same* held-out random rules, with no seed predicate, were translated 8 of 8 correctly by the adapter after the language stage
and by none after the seed stage, which answered "X is encrypted." or wrote world facts instead of rules. The seed stage, with no
translation row, two epochs and the same system prompt, overwrote the language. Run 7 changed exactly one thing — it replays language
rows inside the seed stage — and the language came back completely (12/12), with the verdicts on the package worlds unchanged within
noise; so replay fixes the forgetting. It did not fix the prose: asked to translate the seed, run 7 still answered with a verdict.

**Why run 7 still could not explain the seed** was measured on the same adapter with three variants of the eight explain prompts: an
explicit instruction to answer in sentences gave 2/8 prose answers (as without it); renaming every seed predicate to a neutral word gave
3/8, and that prose was garbled ("If R is zephyr, one must row an S"). Two causes, both supported: *format specialization* — fine-tuning
on one output format makes a model use it for unrelated prompts [75] — reinforced because half of the training's judge prompts showed
the seed's code before the situation, so "seed code → verdict" was literally taught; and a *coverage gap* — the language stage's
templates almost only produced unary and binary predicates over X, Y, Z, while the seed uses arity 3–4, constants as arguments, nested
duties and the letters S, P, O, which is a structural, not a lexical, generalization [76]. Codex, asked independently through the
memory, reached the same leading hypothesis from the literature and added a constraint we adopted: a round-trip reward on the seed
itself would be training on feedback about the withheld target, so GRPO with that reward is used only on synthetic programs. Run 8
addresses both causes; run 8b removes one of them to attribute the effect.

## 7 What went wrong, and what we learned from it

A system built on "evidence or hypothesis" must report its own errors. In the work for this report:

- **An index file's way back created copies.** Recognising one's own content compared only the first 20 objects with the same text; 40 of
  180 contents came back as copies in a real test with many objects titled "What is missing". Fixed (content first, then up to 500), with a
  regression test.
- **Nobody could sign in to the internal search in a real browser.** The pages had inherited `Referrer-Policy: no-referrer`; Chrome then sends
  a form POST with `Origin: null`, and the cross-site check rightly refused it. Unit tests that set the header themselves could not see it.
- **A statistic was misread** (Section 4.6) and **an ordering passed by chance** (Section 5.4).
- **An internal host name entered a commit** although the pre-commit scan had flagged it — the scan's result did not stop the command
  chain. Caught before publishing, removed, and the scan now blocks.
- **The first training data asked each object for its own text** (run 1, 6.5): a falling loss hid a useless model. The adoption gate never
  uses the loss since.
- **A philosophy training erased what it was built on** (run 6). Each stage was measured, but the seed stage's evaluation contained no
  translation task, so the loss of the language stage only showed in the prose test.
- **The logic engine misnamed contradictions** (Codex's review before the seed was built on it): `p. -p. q.` lost the unrelated `q`,
  contradictions of arity ≥ 2 went undetected, a quoted `"-x"` was read as a negation, and an unsatisfiable odd loop was reported as a
  contradiction. Fixed through clingo's own AST.
- **One word named two things.** `widerlegt` meant both "withdrawn by the author" and "refuted by others" (5.8) — a violation of the
  identity rule in the memory itself, found only when a formal seed read the memory.
- **Evaluation worlds leaked into training.** The seed's training worlds are drawn at random, and small worlds recur: 73 training rows of
  run 5 and 12 of runs 6/7 had a world that also stood in their evaluation. Re-scored on clean worlds, run 5 moved from 22/40 to 20/38
  exact, so the conclusions hold, but the first published numbers were too good. Run 8's kit drops every training world that occurs in
  the evaluation and asserts it.
- **A plausible improvement made search worse.** Embedding an object's content together with its key statement in one vector lowered
  Hit@4 from 0.963 to 0.843 on a 5,000-object universe (45 queries worse, 13 better); it was not adopted (4.7).

## 8 Limitations and open problems

- **No human gold set yet.** κ between two prompts measures the model's consistency with itself, not correctness (Pangakis et al. found
  precision or recall below 0.5 on 9 of 27 annotation tasks [41]).
- **Fine-tuning is measured on one small model.** Run 3 injected knowledge into a 0.6B model (7/60 correct against 0/60, p = 0.008) but
  it never abstains; the seed runs teach verdicts but, so far, no model has passed the prose round trip (6.5). Larger models are untested.
- **The seed acts on nothing yet.** The provenance adapter runs in shadow mode; the production memory has almost no cross-instance
  confirmations, so copies, model-made origins and hearsay have not yet occurred there. `untestable`, `extraordinary` and verified
  sources have no source in the memory.
- **The back-translator is one model.** The round trip's control (74 % with Codex) shows its limit; a second independent family and a
  blind human reading are still to come.
- **The total effect of hints is not yet established** (the deployed hold-back has too few observations).
- **Poisoning.** PoisonedRAG succeeds 97 % of the time with five texts per target question [42], AgentPoison above 80 % with under 0.1 %
  poisoned entries [43]. Our own poisoning test (`vi-mcp redteam`, a throw-away memory, success = top 3 of search or shown as a hint)
  found imports and token-less proposals held 10 of 10, but **10 of 10 planted entries of an instance with a token reached an agent**.
  Fixed rules now hold such entries (instructions to an AI, hidden characters, piping a download to a shell, wiping system directories,
  switching a firewall off, handing credentials to a URL): 0 of 10 for the attacks the rules were tuned on, but **6 of 10 for attacks
  written afterwards** — fixed rules close known channels and generalise poorly; a plausible but wrong answer is invisible to them. The
  first version of the rules held 216 of 300 real objects; the version kept holds 1 — measuring false positives on real data was as
  important as measuring attacks.
- **Off-policy training is gated, not proven.** Every training fits an inverse-propensity-weighted router beside the uncorrected one and
  keeps it only if it is better on an inner chronological validation by more than 2 SE (since 0.64.0, the final holdout stays untouched); whether it ever wins on production data is not yet known.
- **Order-dependent tests.** Two tests fail rarely in a full run and pass alone; the cause is unknown.
- **The logic layer's weight counting** approximates LPMLN weight learning where several rules derive the same conclusion. Rule learning
  covers unary conclusions with up to two conditions and has no production data yet (127 conclusions, all right — no negatives).
- **clingo-object's agent is measured on 16 questions, one run each** — a trend, not a proof; whether agents in daily use choose
  `vi_logic answer` over prose and searches is not measured yet.
- **The language does not yet rank better.** Every reordering by question type, label or word weight lowered MRR on the production
  copy (5.4); a relevance weight learned from what is used after a search is the open step.

## 8.1 Implementation audit and virtual machine contract

The 14 November audit reproduced source-page leakage across training/evaluation, action observations split across train/holdout,
challenger metrics displayed under champion weights, and an insufficient non-harm gate. Version 0.64.0 groups training partitions,
separates model reports and requires a noninferiority lower bound. Fine-tuning adoption uses independent correctness or executable
program checks rather than keyword coverage. One correct sampled conclusion no longer certifies a complete generated program.

VIL and clingo-object form the instruction and execution architecture of a persistent virtual intelligence machine (`vi_machine`):
private mutable objects and rules, snapshots, forks, revision guards and reference-checked training examples. The model writes
programs; the runtime controls state and execution. This does not virtualize neural parameters, certify the underlying facts or
establish a fine-tuning gain. See [the executable machine contract](VIRTUAL_MACHINE.md) and its regression tests.

The optional [kNN-LM backend](KNN_LM.md) also implements inference below MCP: a causal model's hidden states index a mutable
next-token datastore, and nearest-neighbor token probabilities are interpolated with its unchanged base distribution. A pinned
pretrained-model experiment changed the next token by replacing one source while preserving the full model-weight hash. This
establishes the mechanism on a memorized context; its held-out perplexity evaluator is available for broader quality measurements.

The [complete instance runtime](INSTANCE_RUNTIME.md) adds context, model/device identity, explicit capabilities and a resumable
client-side execution loop. A coherent local bundle restores this state with its kNN datastore while sharing the hash-matched frozen
model. Pending external actions are not cloned or silently replayed. In an isolated real-model trial the loop read a trusted synthetic
inventory, stored its fact and executed the correct port query in four model calls. This is functional integration evidence rather
than a general-intelligence or before/after model-training result.

The generated training kit additionally audits complete assistant targets with the actual tokenizer before constructing the real
SFT, KTO or DPO trainer. A no-training preflight verifies tokens, target masks and one finite loss with unchanged parameters and zero
optimizer steps. All five supported stages passed on a tiny Transformer fixture and on a tiny initialized Qwen3 architecture with
the real Qwen3 tokenizer. This establishes API/format compatibility for those fixtures, not target-GPU compatibility or a learned
capability gain. See [training preflight](TRAINING_PREFLIGHT.md); the user has deferred actual fine-tuning.

## 9 Conclusion

vi-mcp implements adaptive context selection without changing the language model. Historical randomized comparisons favour ranked hints in some windows; the total benefit of showing hints requires a sufficiently powered independent measurement.
The incubator turns "is this memory good enough?" into criteria with values, and the export turns a mature memory into training data that
another AI can learn from, including how to search the memory itself. VIL keeps the symbols readable for people and checkable for
parsers, and puts meaning where the system already has it: in measured weights, exchanged between memories as measurements rather than
commands. clingo-object adds inference over the memory's own objects in one language for people, agents and the server — and the
measurement suggests that for questions over many objects, a model writing a logic program beats a model reading search results. A philosophy seed carries the same rule set
into an AI's own judgments: it says how well something is supported and where it comes from, counts origins rather than voices, and
names every holding back. The rule set behind it all — evidence or hypothesis, named unknowns, nothing silent, no self-rating — is what
made the errors in our own work visible, including, through the seed, one in the memory's own vocabulary.

---

## References

Sources consulted for this report on 13 November 2026 (web research during the development session); arXiv identifiers and URLs as
retrieved. Where only an abstract was read, the claim cited is the one stated there. Author names are given as commonly cited; they and
reference 1 (Guo et al.) were not re-checked in that session — the identifiers and URLs were.

1. Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
2. Swaminathan, Joachims. The Self-Normalized Estimator for Counterfactual Learning. NeurIPS 2015. (cited in vi_mcp/metrics.py)
3. DroidSpeak: KV cache sharing across LLMs. https://arxiv.org/abs/2411.02820
4. LatentMAS: latent collaboration in multi-agent systems. https://arxiv.org/abs/2511.20639
5. Morris et al. Text Embeddings Reveal (Almost) As Much As Text (vec2text). https://arxiv.org/abs/2310.06816
6. Jha et al. Harnessing the Universal Geometry of Embeddings (vec2vec). https://arxiv.org/abs/2505.12540
7. LoRATK: LoRA once, backdoor everywhere. https://arxiv.org/abs/2403.00108
8. Lewis et al. Deal or No Deal? End-to-End Learning for Negotiation Dialogues. https://arxiv.org/abs/1706.05125
9. Kottur et al. Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog. https://arxiv.org/abs/1706.08502
10. FIPA ACL Message Structure Specification SC00061G. http://www.fipa.org/specs/fipa00061/SC00061G.html
11. FIPA Communicative Act Library Specification SC00037J. http://www.fipa.org/specs/fipa00037/SC00037J.html
12. Wooldridge. Verifiable Semantics for Agent Communication Languages. ICMAS 1998. https://eprints.soton.ac.uk/259563/
13. Singh. Agent Communication Languages: Rethinking the Principles. IEEE Computer 1998. https://www.csc2.ncsu.edu/faculty/mpsingh/papers/mas/computer-acl-98.pdf
14. Debenedetti et al. Defeating Prompt Injections by Design (CaMeL). https://arxiv.org/abs/2503.18813
15. OWASP Top 10 for LLM Applications, LLM01 Prompt Injection. https://genai.owasp.org/llmrisk/llm01-prompt-injection/
16. Li, Roth. Learning Question Classifiers. COLING 2002. https://aclanthology.org/C02-1150
17. de Haan. Coding of Evidentiality. WALS ch. 78. https://wals.info/chapter/78 — and Bickel, Nichols, Fusion; WALS ch. 20. https://wals.info/chapter/20
18. Dryer. Order of Subject, Object and Verb. WALS ch. 81. https://wals.info/chapter/81
19. Elastic. Text-to-ES|QL benchmark. https://www.elastic.co/search-labs/blog/text-to-esql-benchmark
20. CypherBench. https://arxiv.org/html/2412.18702v2
21. Elasticsearch reference: query_string query. https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html
22. The Complete Lojban Language, ch. 13.11 Evidentials. https://lojban.org/publications/refgram_chunked/cllc/13/11/
23. Sumers et al. Cognitive Architectures for Language Agents (CoALA). https://arxiv.org/abs/2309.02427
24. Pezeshkpour, Hruschka. Large Language Models Sensitivity to the Order of Options. Findings NAACL 2024. https://aclanthology.org/2024.findings-naacl.130
25. Reiss. Testing the Reliability of ChatGPT for Text Annotation. https://arxiv.org/abs/2304.11085
26. Xiong et al. Can LLMs Express Their Uncertainty? ICLR 2024. https://arxiv.org/abs/2306.13063
27. docTTTTTquery (doc2query). https://github.com/castorini/docTTTTTquery
28. Krippendorff's alpha, decision thresholds. https://www.statisticshowto.com/krippendorffs-alpha/
29. Pangakis, Wolken. Knowledge Distillation in Automated Annotation. NLP+CSS 2024. https://aclanthology.org/2024.nlpcss-1.9/
30. Penedo et al. The FineWeb Datasets. https://arxiv.org/abs/2406.17557
31. Gebru et al. Datasheets for Datasets. https://arxiv.org/abs/1803.09010
32. Ethayarajh et al. KTO: Model Alignment as Prospect Theoretic Optimization. https://arxiv.org/abs/2402.01306
33. Rafailov et al. Direct Preference Optimization. https://arxiv.org/abs/2305.18290
34. Chen et al. FireAct: Toward Language Agent Fine-tuning. https://arxiv.org/abs/2310.05915
35. Zeng et al. AgentTuning. https://arxiv.org/abs/2310.12823
36. Data contamination in LLM evaluation. https://arxiv.org/abs/2310.18018
37. Hugging Face TRL, dataset formats. https://huggingface.co/docs/trl/main/en/dataset_formats
38. Dettmers et al. QLoRA. https://arxiv.org/abs/2305.14314
39. llama.cpp server: LoRA adapters. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
40. Unsloth requirements (VRAM table). https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements
41. Pangakis et al. Automated Annotation with Generative AI Requires Validation. https://arxiv.org/abs/2306.00176
42. Zou et al. PoisonedRAG. https://arxiv.org/abs/2402.07867
43. Chen et al. AgentPoison. https://arxiv.org/abs/2407.12784
44. Lifschitz. What Is Answer Set Programming? https://www.cs.utexas.edu/~vl/papers/wiasp.pdf
45. Stanford Encyclopedia of Philosophy. Deontic Logic. https://plato.stanford.edu/entries/logic-deontic/
46. Lee, Wang. Weight Learning in a Probabilistic Extension of Answer Set Programs (KR 2018). https://arxiv.org/abs/1808.04527
47. Hahn et al. plingo: A System for Probabilistic Reasoning in Clingo Based on LPMLN. https://arxiv.org/abs/2206.11515
48. Yang, Ishay, Lee. NeurASP: Embracing Neural Networks into Answer Set Programming (IJCAI 2020). https://www.ijcai.org/proceedings/2020/243
49. Law, Russo, Broda. The ILASP System for Inductive Learning of Answer Set Programs. https://arxiv.org/abs/2005.00904
50. Potassco. clingo (Python package `clingo`). https://pypi.org/project/clingo/
51. Upadhyay et al. UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor. https://arxiv.org/abs/2406.06519
52. Wang et al. Position Bias Estimation for Unbiased Learning to Rank in Personal Search (WSDM 2018). https://research.google/pubs/position-bias-estimation-for-unbiased-learning-to-rank-in-personal-search/
53. Szeider. ASP-Bench: natural-language problems for answer set programming, with an agentic study (2026). https://arxiv.org/abs/2602.01171
54. ALM–ASP: language agents for answer set programming modelling (KR 2026). https://proceedings.kr.org/2026/103/
55. DLV-DB user manual (#import / #export between relational tables and predicates). https://www.dlvsystem.it/dlvsite/dlvdb-user-manual/
56. Ishay, Yang, Lee. Leveraging Large Language Models to Generate Answer Set Programs (KR 2023). https://arxiv.org/abs/2307.07699

Added on 15 November 2026 (web research for sections 4.7, 5.7, 5.8 and 6.5; identifiers and URLs as retrieved, claims as stated
in the abstract or summary read):

56b. Allen-Zhu, Li. Physics of Language Models: Knowledge Storage and Extraction. https://arxiv.org/abs/2309.14316
56c. Yang et al. Synthetic continued pretraining (EntiGraph). https://arxiv.org/abs/2409.07431
56d. Gekhman et al. Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? https://arxiv.org/abs/2405.05904
57. Olausson et al. LINC: logical inference via neurosymbolic computation. https://arxiv.org/abs/2310.15164
58. Pan et al. Logic-LM. https://arxiv.org/abs/2305.12295
59. AbstentionBench. https://arxiv.org/abs/2506.09038
60. Solver-filtered SFT vs DPO for answer set programs (clingo 5.7.1, 8B models). https://arxiv.org/abs/2512.17093
61. TruthRL: a ternary reward (correct / abstain / hallucinate). https://arxiv.org/abs/2509.25760
62. GRPO on small models (Qwen2.5-0.5B). https://arxiv.org/abs/2503.18892 — and DeepSeek-R1, rule-based rewards. https://arxiv.org/abs/2501.12948
63. FormaRL: a compile-only reward is gamed. https://neurips.cc/virtual/2025/130519
64. Jury theorem. https://en.wikipedia.org/wiki/Jury_theorem
65. Spiekermann. Jury theorems. https://researchonline.lse.ac.uk/id/eprint/101162/1/Spiekermann_Jury_theorems.pdf
66. Bikhchandani, Hirshleifer, Welch. A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades. JPE 100(5), 1992. https://snap.stanford.edu/class/cs224w-readings/bikhchandani92fads.pdf
67. Wojcik et al. Birdwatch: Crowd Wisdom and Bridging Algorithms can Inform Understanding and Reduce the Spread of Misinformation. https://arxiv.org/abs/2210.15723
68. Shumailov et al. AI models collapse when trained on recursively generated data. Nature 631, 2024. https://www.pure.ed.ac.uk/ws/portalfiles/portal/460496122/ShumailovEtalNature2024AIModelsCollapseWhen.pdf
69. Stanford Encyclopedia of Philosophy. The Pragmatic Theory of Truth. https://plato.stanford.edu/entries/truth-pragmatic/
70. Stanford Encyclopedia of Philosophy. Epistemological Problems of Testimony. https://plato.stanford.edu/archives/spr2012/entries/testimony-episprob/
71. C2PA Technical Specification 2.2. https://c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
72. Prelec, Seung, McCoy. A solution to the single-question crowd wisdom problem. Nature 541, 2017. https://news.mit.edu/2017/algorithm-better-wisdom-crowds-0125
73. Snowball stop word lists (German, English). https://snowballstem.org/algorithms/german/stop.txt · https://snowballstem.org/algorithms/english/stop.txt
75. Wang et al. Two-stage LLM Fine-tuning with Less Specialization and More Generalization (format specialization, ProMoT). https://arxiv.org/abs/2211.00635
76. Kim, Linzen. COGS: A Compositional Generalization Challenge Based on Semantic Interpretation. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.731/
77. Berglund et al. The Reversal Curse. https://arxiv.org/abs/2309.12288 · Berglund et al. Taken out of context. https://arxiv.org/abs/2309.00667 · Betley et al. Tell me about yourself. https://arxiv.org/abs/2501.11120 · Treutlein et al. Connecting the dots. https://arxiv.org/abs/2406.14546
74. eclingo (epistemic logic programs). https://arxiv.org/abs/2008.02018 · ASPARTIX (argumentation on clingo). https://arxiv.org/abs/1507.06689 · deolingo (deontic ASP). https://www.dc.fi.udc.es/~cabalar/deolingo.pdf

Further sources used in the design notes of the same session: Gilardi et al., ChatGPT outperforms crowd workers (https://arxiv.org/abs/2303.15056);
Ziems et al., LLMs for computational social science (https://arxiv.org/abs/2305.03514); Agora protocol (https://arxiv.org/html/2410.11905v1);
Spotlighting (https://arxiv.org/abs/2403.14720); Zep/Graphiti (https://arxiv.org/abs/2501.13956); MemoryBank (https://arxiv.org/abs/2305.10250);
A2A specification (https://a2a-protocol.org/latest/specification/); MCP specification, tools (https://modelcontextprotocol.io/specification/2025-06-18/server/tools).

## Appendix A — Measurements cited

| quantity | value | when, how |
|---|---|---|
| knowledge objects | 24,379 | production, `vi_stats`, 13.11.2026 |
| router | AUC 0.863–0.865 on 17,590 observations | production, model report, 13.11.2026 |
| search MRR (fixed set, 150 × 2) | hybrid 0.735 / 0.732; full text 0.364 / 0.432 | daily measurement 12./13.11.2026 |
| exploration − ranked | −6.4 ± 3.7 / −6.9 ± 3.4 points | daily measurement 12./13.11.2026 (7 days each) |
| warnings | success after 51 %, without 98 % | `vi_stats`, 24 h to 13.11.2026 |
| empty searches | 111 of 424, all monitoring lookups of 57 unknown hosts | production search log, read only, 13.11.2026 |
| LLM label agreement | type 86.3 % (κ 0.80), memory 92.3 % (κ 0.88), n = 300 | stratified production sample, 13.11.2026 |
| answers outside the schema | 28 of 600 (4.7 %) | same run |
| student (C = 10, sure ≥ 0.8) | type 94.4 % right on 15.5 %; memory 95.0 % on 24.6 % | 252 agreed labels, 5-fold CV |
| index file round trip | 185 objects; export 180 in 0.25 s (218 KB); import 0.78 s; way back 0 copies (40 before the fix) | two local test servers, 13.11.2026 |
| LLM weight layer, offline | log loss −0.00006 ± 0.00011 / observation, 3,645 holdout | production, 13.11.2026 |
| clingo-object vs SQL | 13/13 equal at 20,000 objects (3/13 short at 2,000); 0.01–0.6 s, 2.3 s for 13,088 objects | production copy, 13.11.2026 |
| clingo-object agent | 15/16 exact, held out 4/4, P 1.00, R 0.99, 5.2 s; `vi_recall` alone 2/16, P 0.50, R 0.30, 20.1 s | production copy, 16 questions, one run, 13.11.2026 |
| agent on production | "DMZ hosts never described as production": 2 hosts, 1 round, 5.3 s | `vi_logic action=answer`, 0.63.0, 14.11.2026 |
| seeded training data | 16 rule sets equal to SQL on live data; 127 conclusions rated, all right; 14 rule weights | production, 14.11.2026 |
| search Hit@4 (hybrid) | 0.933 production copy; 0.930 at 500,310 objects | `measure.py`, 300 queries, 15.11.2026 |
| content in the one vector | Hit@4 0.963 → 0.843 (45 worse, 13 better) | 5,000-object universe, same queries, 15.11.2026 |
| stop words out of the query | full text, questions, held-out half: Hit@4 0.347 → 0.533 (16/2, p 0.0013); hybrid 0.933 → 0.947 (p 0.22) | production copy, split by target before any run, 15.11.2026 |
| round trip control | Codex rebuilt the seed from reference prose: 49/66 worlds (74 %), 1 round | `philosophy_explain.round_trip`, 15.11.2026 |
| run 5 (seed) | clean worlds 20/38 exact, paraphrase 18/37, pressure 15/26, formalize 13/20; base 0 (with leaked worlds: 22/40) | `phil_score.py` against the seed, worlds of its training removed, 15.11.2026 |
| run 7 (replay) | random rules 12/12; clean worlds 19/40, paraphrase 4/37, pressure 7/30; seed prose 2/10 | `phil_score.py`, `probe7.py`, 15.11.2026 |
| why no prose (run 7) | plain 2/8, with format instruction 2/8, predicates renamed 3/8 (garbled) | `probe_route.py`, 15.11.2026 |
| training speed | 240 steps: RTX 3070 43.6 s fp32 / 46.4 s bf16; Strix Halo 64 s fp32 | same kit, 15.11.2026 |
| run 6 (forgetting) | random rules translated 8/8 after the language stage, 0/8 after the seed stage | `probe_ab.py`, same items, two adapters, 15.11.2026 |
| seed vs memory (shadow) | 24,479 / 24,499 agree before P13, 24,493 after; 16 of 17 "refuted" objects refuted only by their author | production copy, read-only, 15.11.2026 |

## Appendix B — Glossary (stored names are German)

| stored | meaning |
|---|---|
| erkenntnis, regel, erfahrung, gespraech | insight, rule, experience, conversation |
| geteilt, intern, privat | shared, internal, private |
| thema, fehler, art_wissen | topic, error, kind of knowledge |
| schlaf, schueler, einig/uneinig | sleep, student, agreed/disagreed |
| inkubator: erfuellt, offen, nicht_gemessen, reif, im_inkubator | met, open, not measured, mature, in the incubator |
| widerlegt, zurueckgezogen, strittig | refuted (by two foreign model families), withdrawn (by the author), disputed (by one foreign family) |
| grund | reason (the seed's reason predicate) |
