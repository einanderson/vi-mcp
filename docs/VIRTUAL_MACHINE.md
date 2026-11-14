# Virtual intelligence machine

`vi_machine` (0.64.0) executes VIL over persistent, mutable objects and clingo-object rules. The LLM writes programs; the runtime
owns their input world, executes the safe notation, records results and compares complete answers with supplied references.
This is an executable first machine contract, `vil-machine-v1`. It does not emulate a neural network's weights or promise identical
natural-language behavior across models.

The [complete instance runtime](INSTANCE_RUNTIME.md), `vil-instance-v1`, extends this world with context, model/device bindings,
permissions and a controlled client-side loop. Its bundle checkpoint captures the symbolic world and the local kNN datastore together.

## The machine contract

| Component | Implementation |
|---|---|
| Instructions | VIL acts and structured operands through `vi_machine` |
| Program representation | clingo-object safe notation; Python classes compile to this same representation |
| Execution | clingo, with the notation's named true/false/both/unknown/undetermined states and constraint reports |
| Mutable world | Objects, tags and saved rules, private to one named machine and authenticated instance |
| Persistence | Private `maschine` objects in the existing database; no second writer process |
| Concurrency | Atomic transitions under the database lock; optional `expected_revision`; a model call runs outside the lock and commits only if its input revision is unchanged |
| Snapshots | Immutable copies of the world, addressed by owner and snapshot name |
| Fork | Independent copy of the current world or a snapshot, under the same authenticated owner |
| Verification | Entire `ans(...)` answer set against an explicitly supplied independent reference; missing and extra answers both matter |
| Learning | Explicit publication of checked examples as internal knowledge; SFT, KTO and same-task DPO export |

`vi_mcp/machine.py` implements the lifecycle. `clingo_object/notation.py` implements execution, and `clingo_object/filters.py`
shares filter semantics between live knowledge, frozen machine worlds and training evaluation. A machine never searches global
knowledge implicitly. `import` copies only objects visible to the caller; repeating it refreshes that mounted set. Objects no
longer present in the mounted set leave tombstones. Other machines' worlds do not change.

The execution state hash covers the world, rules and imported sets. A revision increases on committed instructions, including reads
that create execution receipts. A restore reinstates the world hash and advances the revision; it preserves the audit history.
A fork copies the world and starts its own history. Model parameters, an external provider's KV cache, physical time and external
systems are outside this snapshot contract. The frontend uses the server's configured classification model, with the existing
provider fallback. Changing that model can change generated programs; execution of the same program over the same frozen world
and rule priors retains the clingo-object semantics.

The optional [kNN-LM inference backend](KNN_LM.md) additionally changes token predictions through a mutable hidden-state/next-token
datastore without training the base model. Standalone datastore snapshots remain available; the instance host coordinates them with
the complete symbolic state when creating a bundle checkpoint.

## Execute and change facts

Calls below are MCP tool calls. They can also be issued with `vi-mcp machine ACTION NAME`, which connects through the normal MCP
client configuration and never opens the server database directly. `--payload FILE` reads structured operands from a JSON file.

```python
vi_machine(action="create", name="operations")
vi_machine(action="execute", name="operations", instruction="tell.fact api",
           payload={"text": "API endpoint", "hashtags": ["service:api", "port:80"], "source": "checked inventory"})
vi_machine(action="execute", name="operations", instruction="def.rule ports",
           payload={"source": 'ans(P) <= (O in objects("service:api")) & (O.port == P)'})
vi_machine(action="execute", name="operations", instruction="ask.rule ports")
# answers: ans("80")

vi_machine(action="snapshot", name="operations", snapshot="before-change")
vi_machine(action="execute", name="operations", instruction="tell.fact api",
           payload={"text": "API endpoint", "hashtags": ["service:api", "port:8080"], "source": "checked inventory update"})
vi_machine(action="execute", name="operations", instruction="ask.rule ports")
# same program, new answer: ans("8080")

vi_machine(action="fork", name="operations", target="experiment", snapshot="before-change")
vi_machine(action="restore", name="operations", snapshot="before-change")
```

`tell.fact NAME` replaces that object's complete tag set, so an old port does not remain beside the new one. `tell.object` is an
alias. `payload.deleted=true` leaves a tombstone (give `reason`). Facts require a source. `ask.what FILTER` reads the machine's
objects. `ask.rule NAME` executes the saved rules together; `payload.source` adds an ephemeral program. `def.rule NAME` validates and
stores safe notation. Rules use their declared soft-rule priors in this isolated contract; changing live VI rule weights does not
silently change an existing machine.

`do` and `promise` create recorded proposals. They never invoke a shell, another MCP service or an external write. A proposal can be
inspected in its receipt; this contract does not implement approval and execution of external side effects. Unknown instructions,
unsupported rules and stale revisions return reasons. Raw callers' Python is never executed.

## Let an LLM write the program

```python
vi_machine(action="answer", name="operations", question="Which port does the API use?")
```

The frontend receives only this machine's vocabulary, tries up to four programs and gets the actual solver refusal or result between
attempts. Saved machine rules are included during both checking and final execution. The returned program, complete projected
answers, world, revision and execution receipt are retained. Generated code cannot write the world. Only explicit VIL instructions
change it. A model call racing with a world change is refused at commit, rather than returning an answer under a different revision.

## Teach the model to program the machine

```python
vi_machine(action="execute", name="operations", instruction="rate RECEIPT",
           payload={"question": "Which port does the API use?", "expected_answers": ['ans("80")'],
                    "reference": "independent inventory check"})
vi_machine(action="teach", name="operations", receipt="RECEIPT")
```

Use the execution receipt returned by `answer` or `ask.rule`. Supply the entire reference answer set; an empty list is a valid
reference. `rate` checks against the frozen execution, not against a later world. A timeout, ambiguous answer or truncated execution
cannot become a positive example. The reference is a caller-supplied assertion whose origin is recorded; it is not certified by an
LLM or by clingo. An independently implemented SQL query, inventory fixture or human review can supply it.

Private machines and snapshots are excluded from exports. `teach` is an explicit promotion of one checked example to internal
knowledge, including its frozen inputs. The owner must intend to share those inputs within the organisation. It never publishes to
an open network. It refuses when the execution read an imported object that is private or not knowledge (experience, conversation),
and it publishes the private instance context only with `include_context=true`. The export audits credential scrubbing again.

The training pipeline uses the checked evidence as follows:

1. A correct program gives a tool-use SFT example in `logic.jsonl` with its actual machine inputs and outputs.
2. An incorrect program gives a KTO negative. Earlier repair attempts are replayed against the same reference before being labelled.
3. Correct and incorrect programs for exactly the same prompt and world give DPO pairs. A shared hashtag alone is not a verified preference.
4. The kit partitions all stages together by object, source, duplicate content and task/program family. Solver-only demonstrations
   and observational hint pairs are retained in the export for inspection but withheld from the default program/DPO training splits.
5. `program_evaluate.py` executes generated programs against frozen held-out worlds. Extra answers, missing answers, syntax errors,
   contradictions, incomplete enumeration and inappropriate constraint states count as failures. It accepts cached predictions for
   offline replay. `program_eval.jsonl` contains only tasks assigned to the test side of the shared split.

`vi_logic(action="verify", source=..., question=..., expected_answers=[...], reference=...)` supplies the equivalent complete-answer
check for programs over the live VI. It ties evidence to a hash of imported facts, compiled program, solver version and soft weights.
An old reference is not reused after that execution state changes. A few individually rated conclusions cannot certify a whole
natural-language question or a program's completeness.

The kit's ordinary `evaluate.py` collects paired predictions and requires independent correctness judgments. Keyword overlap no
longer authorizes adoption. `record` accepts only reports bound to the held-out file with a positive conservative lower confidence
bound, calculated over independent task/source groups. No training is started by the VI.

## Limits and reproducible checks

Contract v1 limits: 500 objects including tombstones, 16 snapshots per machine, 16 execution receipts and checked examples, the
existing object-data size ceiling and notation resource limits. Dropped old receipts are counted. A full limit is refused with a
reason. Snapshot names are unique within an owner. Edges are not implemented in machine v1; join objects by their tag values.

The runtime has one execution backend (clingo). Model frontends can be replaced; this is not evidence that multiple independent
execution backends conform. Fine-tuning improvement requires an actual before/after run on held-out tasks. Formal execution cannot
prove that source facts are true or that a natural-language request was translated correctly.

```bash
.venv/bin/python scripts/demo_machine.py
.venv/bin/python -m pytest -q tests/test_machine.py tests/test_training_audit.py -p no:cacheprovider
python3 scripts/gen_reference.py --check
```

Research context: [MemGPT](https://arxiv.org/abs/2310.08560) develops virtual context management;
[ReST-EM](https://arxiv.org/abs/2312.06585) filters generated solutions for self-training;
[Solver-in-the-Loop for ASP](https://arxiv.org/abs/2512.17093) constructs solver-guided training examples;
[ASP-Bench](https://arxiv.org/abs/2602.01171) includes independent validators of the intended specifications.
