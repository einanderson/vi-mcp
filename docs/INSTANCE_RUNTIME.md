# Complete virtual instance runtime

`vil-instance-v1` adds the full instance around the [VIL machine](VIRTUAL_MACHINE.md): a shared model core, its own mutable VI world,
persisted context, executable rules, explicit MCP capabilities and a resumable execution loop. With the local [kNN backend](KNN_LM.md),
one checkpoint captures both the symbolic instance and its neural token datastore. The base weights are shared and verified by hash.

| Virtual machine component | Implemented instance component |
|---|---|
| Computation core | Shared frozen causal model; model, weight, tokenizer, configuration and decoder identities |
| Working memory | Persisted role-labelled context, goal and execution state |
| Persistent storage | Private VI objects, facts, imported knowledge, rules and experience receipts |
| Programs | Safe clingo-object notation and VIL instructions |
| Devices | Trusted client-side adapters for named MCP tools |
| Permissions | An explicit per-instance allow-list; every listed device must exist unchanged on the current host, otherwise boot and run are refused (never silently narrowed) |
| Execution loop | Model plan → fixed validation → instruction/device → observed feedback → next plan |
| Snapshot and clone | Private instance bundle plus immutable kNN shards; a new datastore and VI world on boot |

`clingo_object/runtime.py` is a standard-library client runtime; it never imports vi_mcp. `vi_mcp/machine_runtime.py` validates the
declarative state on the server. `vi_mcp/machine_host.py` supplies the CLI and optional local neural adapter. Neither an MCP caller
nor the language model can send Python to be executed on the memory server.

## State and control

`vi_machine action=inspect` returns the authenticated owner's complete world. `configure` replaces its runtime descriptor; `context`
appends messages, or explicitly replaces them with a compaction/deletion reason. Objects, rules, context, model binding, permissions
and loop state participate in the state hash and snapshots. Old machines acquire the extra fields when explicitly configured or
booted; their previous receipts retain their original state hashes.

The runtime descriptor has `contract`, `model`, `devices`, `permissions` and `max_steps`. A device descriptor contains a stable
`binding`, MCP `tool`, `effect` (`read` or `write`), argument `schema` and `description`. Actual endpoint URLs and credential files
belong to the trusted host profile; they are not serialized into a snapshot. An unspecified device credential means unauthenticated,
never “send the VI's default token to this other server”. Registering a device alone grants no permission to invoke it.
A device never names a VI owner operation (`vi_machine`, `vi_release`, `vi_quarantine`, `vi_set_visibility`, `vi_net_token`,
`vi_net_switch`, `vi_tools_register` and their German aliases, `clingo_object.runtime.OWNER_TOOLS`): through it the model could
reconfigure its own rights or release its own proposals. Host and server refuse such a descriptor.

The model sees the instance world and context as data and may request an instruction, a registered device or completion. It cannot
request permission changes, model replacement, independent reference ratings, publishing or snapshot control. Fixed validators
check its JSON plan and device arguments. The supported schema subset covers object properties, required fields, additional
properties, arrays, primitive types, enums, bounds and patterns. Unsupported schema validation keywords are refused.

The controller asks `vi_before` before an actual device call and reports `vi_after` afterwards. Hints and reflex suggestions do not
expand capabilities or silently replace the planned arguments. Hint counts are retained without asserting that a hint helped.
Telemetry failure is recorded and never authorizes repeating the external action. The subsequent model step receives the observed
result. Tool text cannot grant permission or change a transport binding.

## Reserve and observe external actions

The server persists a reservation before the trusted host invokes a device. It records the exact arguments, descriptor and policy
hash. A completed observation consumes that reservation once. A failed read can become ordinary repair feedback; a write whose
transport fails becomes uncertain. A killed client leaves its prepared reservation pending. Neither is automatically replayed.

Pending actions prevent checkpoints, forks and world replacement. The owner must inspect the external system and explicitly resolve
the reservation as `completed`, `not_run` or `failed`, with a reference:

```python
vi_machine(action="loop", name="operations", payload={"operation": "resolve", "reservation": "RESERVATION_ID",
           "outcome": "completed", "reference": "independent audit of the target system"})
```

The instance remains paused until explicitly resumed. A permission change cannot undo an action already dispatched; restored
permissions cannot restore a capability absent from today's host allow-list. The runtime does not claim transactional rollback of
external systems or exactly-once delivery across network failure.

`finish` ends the loop with `evidence=model_report` and `verified=false`. An agent's completion claim is not an independent grade.
Complete solver references still use `rate`/`verify`; checked examples require explicit `teach` before training export. A corrected
reference can update an already taught example, retaining its object history.

## Run from a trusted host

Use the neural environment described in [KNN_LM](KNN_LM.md), with a compatible instruction model. Save a private profile as
`.vi-instance/bindings.json`; that directory is gitignored. Replace `MODEL`, `COMMIT` and `VI_TOKEN_FILE` with the operator's actual
model identity, pinned model revision and credential path:

```json
{
  "model": {
    "kind": "knn", "model": "MODEL", "revision": "COMMIT", "device": "cuda",
    "store": ".vi-instance/token-memory", "weight": 0.5, "neighbors": 32, "steps": 1024
  },
  "devices": {
    "vocabulary": {
      "binding": "vi-memory-v1", "url": "http://127.0.0.1:8765/mcp", "token_file": "VI_TOKEN_FILE",
      "tool": "vi_vocabulary", "effect": "read", "description": "Read the memory vocabulary",
      "schema": {"type": "object", "properties": {}, "additionalProperties": false}
    }
  },
  "permissions": ["vocabulary"],
  "max_steps": 8
}
```

```bash
.venv-knn/bin/python -m vi_mcp machine run operations --bindings .vi-instance/bindings.json --create --goal 'Read the memory vocabulary'
.venv-knn/bin/python -m vi_mcp machine checkpoint operations --bindings .vi-instance/bindings.json --directory .vi-instance/snapshot --snapshot first-state
.venv-knn/bin/python -m vi_mcp machine boot experiment --bindings .vi-instance/clone-bindings.json --directory .vi-instance/snapshot
```

For boot, `clone-bindings.json` names the same base model and today's allowed devices, but a new, nonexistent `model.store` directory.
The checkpoint's decoder settings and neural datastore are restored there; the original instance keeps its own writable store.
Python callers may share one already loaded `TransformersCore` across several `KNNProgrammer` instances. Their forward-pass lock is
shared as well, so independent datastores cannot race mutable implementation buffers in the common model.

For a text-only inference API, use `model={"kind":"remote","binding":"NAME","model":"MODEL","url":"URL"}` and optional
`token_file`/`identity` fields. Such a bundle records its model binding; it cannot capture an opaque provider's hidden state or cache.
For a local kNN bundle, all evolving state used by this backend is restored: context and datastore. The backend recomputes causal
states from context and does not maintain a separate persistent KV cache. Base weights must be available and match the saved hashes.

## Checkpoint consistency and resource limits

A checkpoint holds the kNN datastore writer lock while the memory atomically snapshots its world at the expected revision. Writers
cannot advance the datastore across that capture. Immutable shards and the world are written into a private bundle; transport
credentials stay on the host. Execution receipts and independent checks are captured as well, so a restored instance can still
teach a previously checked experience. A complete-instance hash covers world, evidence and inference descriptor. Boot verifies these
hashes, inference manifest, model/decoder identity, and current capabilities.
The symbolic snapshot and its parent's revision commit in one database transaction; a rejected parent leaves no orphan snapshot.

A failed capture leaves an explicitly incomplete artifact that boot refuses. Checkpoints require a quiescent device boundary.
They neither copy external devices nor roll back physical time. Snapshot files are private data, not an automatically anonymized
network export. SHA-256 detects corruption; local bundles are not signed attestations from an independent party.

Limits are explicit: at most 16 registered devices, 64 steps per run, 32 context messages/24,000 serialized context characters,
16 retained loop events, and the existing machine/object-data limits. Context compaction needs a reason and counts removed messages.
Old loop events are counted when dropped. Large tool observations retain a marked prefix, full length and hash; they are not treated
as complete knowledge. Device dispatch reserves space for its bounded observation before invoking the tool.

## Verified behavior

`tests/test_instance.py` exercises real MCP transport, the operator CLI, capability refusal, syntax repair, context persistence,
checkpoint/boot, model mismatch, transaction rollback and uncertain-write recovery after reopening the database. The privacy test in
`tests/test_anonym.py` checks that a private bundle preserves its owner's context while omitting endpoint/credential bindings.

`tests/test_instance_neural.py` uses actual frozen causal transformers: boot restores context, rules, permissions and a previous kNN
token prediction while the original datastore changes independently. A second case stores a complete clingo-object program in
neural token memory. It is then generated byte-for-byte, stops at EOS and answers ports 80 and 8080 on two input worlds at unchanged
weight hash. The byte-vocabulary network is a deterministic contract fixture, not a pretrained-model quality benchmark.

An additional isolated run with the configured real LLM performed inventory lookup, fact insertion, clingo program execution and
completion in four model calls and one device call. Its solver answer `ans("8080")` matched an independent synthetic inventory
fixture. Production memory and service configuration were not changed. This verifies that loop, model and device paths compose;
general capability, causal production benefit and fine-tuning gains still require their respective independent measurements.
