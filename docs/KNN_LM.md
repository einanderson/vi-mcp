# Mutable token memory with kNN LM

`vi-mcp knn` (0.64.0) implements kNN-LM inside autoregressive inference. A frozen causal model supplies both its last hidden context
state and its next-token logits. The datastore contains `(context state, following token)` pairs. Squared-L2 nearest neighbors give
a token distribution, which is interpolated with the model's probabilities:

```text
p(token | context) = (1 − weight) × p_base(token | context) + weight × p_neighbors(token | context)
p_neighbors(token | context) ∝ sum(exp(−squared_distance / retrieval_temperature)) for neighbors with that target token
```

This follows the mechanism of [Khandelwal et al., Generalization through Memorization](https://arxiv.org/abs/1911.00172), with a
last-layer key function and exact search. It is an implementation of the method, not a numerical reproduction of the paper's model,
final-feedforward-input keys or FAISS benchmark. Text retrieved through MCP is not appended to a prompt to simulate this mechanism.

## Run the inference backend

Run from the repository root on the inference host. The ordinary memory daemon does not need PyTorch. Create a separate environment;
for a CPU installation, the [Transformers installation instructions](https://huggingface.co/docs/transformers/installation) give:

```bash
python3 -m venv .venv-knn
.venv-knn/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-knn/bin/python -m pip install transformers numpy
```

On a GPU host install the appropriate PyTorch GPU build instead. Models must expose causal-LM logits and hidden states through
Transformers. Model and tokenizer loading use `trust_remote_code=False`. This backend loads actual model weights; a remote chat API
that returns only text cannot supply its hidden states. GGUF/llama.cpp is not this implementation's execution backend.

Approved experience goes into a JSONL file, for example `experience.jsonl`:

```json
{"source":"checked-inventory","prompt":"The access code is","continuation":" blue"}
```

Only continuation tokens become targets. Their keys are the actual preceding causal context states. Choose a tokenizer boundary
between prompt and continuation; an ambiguous boundary is refused. Oversized contexts are refused rather than silently truncated.
Each source is replaced atomically, including all its earlier token entries. No optimizer or training loop runs.

```bash
.venv-knn/bin/python -m vi_mcp knn add --store token-memory --model sshleifer/tiny-gpt2 --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be --data experience.jsonl
.venv-knn/bin/python -m vi_mcp knn predict --store token-memory --model sshleifer/tiny-gpt2 --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be --prompt 'The access code is' --weight 0.75 --neighbors 1
.venv-knn/bin/python -m vi_mcp knn snapshot --store token-memory --destination token-memory-snapshot
.venv-knn/bin/python -m vi_mcp knn forget --store token-memory --source checked-inventory --reason 'inventory no longer applies'
```

Replace `sshleifer/tiny-gpt2` and its pinned revision with a compatible real model for useful generation. The tiny model demonstrates
the mechanism. A larger model needs sufficient memory for inference and extracting context states. The weight hash is computed over
all loaded model parameters and buffers. Model identity, weights, configuration, tokenizer, key function, dimension and vocabulary
must match the datastore exactly. A changed model requires rebuilding its context-state keys.

The neural tests and pretrained experiment ran with Python 3.11, PyTorch 2.15.0+cpu, Transformers 5.23.0 and NumPy 2.4.6 in an
isolated environment. The regular server suite runs without these optional neural dependencies.

## Connect the VI to token inference

```bash
.venv-knn/bin/python -m vi_mcp knn serve --store token-memory --model MODEL --revision COMMIT --device cuda --weight 0.5 --neighbors 32 --bind 127.0.0.1 --port 8776
```

The non-streaming `/v1/chat/completions` endpoint accepts the VI's existing temperature-zero requests, including its thinking-off
field. It renders the model's chat template with thinking disabled. `/v1/completions` accepts plain prompts, `/v1/models` names the
loaded model, and `/health` reports the active revision. The service serializes generation; updates through another `knn add` process
become visible to the next request without restarting or reloading the model. One generation pins its input datastore revision.

Set the VI's configured `classify.url` to the inference service's `/v1/chat/completions` URL and `classify.model` to its loaded model
identifier. `vi_logic answer` and `vi_machine answer` then use the kNN-LM backend through the existing model client. If classification
already uses a separate model, switching this setting also switches classification; choose the backend deliberately. The existing
`token` field carries its bearer credential when configured. A non-loopback listener requires `--token-file`; deployments can place
their existing reverse proxy in front of it.

Model responses include the pinned datastore revision and hash, interpolation parameters and frozen weight hash. Machine execution
receipts retain these in `frontend_model.knn`. The datastore supplies model behavior; VIL and clingo-object still govern execution.
Changing token memory does not authorize a generated program to invoke a shell or foreign service.

`--steps` is an explicit decoding resource budget. Exhaustion or a full context is returned as a length-limited response, which the VI
model client refuses as incomplete. The backend supports greedy text generation; streaming, model-native tool calls and sampling are
refused with a reason. Its compatibility covers the VI text-program frontend, not the entire OpenAI API.

## Persistence and isolation

Each datastore is a private inference-host directory. Immutable NumPy shards hold vectors and integer token targets; a manifest is
atomically replaced under a process/file lock. Readers see a coherent revision. Source replacement and withdrawal leave tombstones;
old shards remain available to pinned requests and snapshots. A snapshot copies one coherent revision into an independent datastore.
There is no VI database connection, implicit network export or automatic ingestion of unreviewed knowledge.

Several `KNNLanguageModel` objects may share one `TransformersCore` while using separate datastores; each HTTP process serves one
datastore. Standalone symbolic and datastore snapshots are explicit operations; the [instance host](INSTANCE_RUNTIME.md) additionally
coordinates them into one private bootable bundle with context, model identity and capabilities. Neither captures an opaque provider's
KV cache or implements Titans. Local inference recomputes its causal states from the restored context.

The default ceiling is one million active entries. Exact search is linear in entries and dimension and operates in blocks of 1,024
vectors. It is suitable for correctness checks and bounded stores; billion-token workloads require an indexed backend and measured
capacity planning. Historical shards consume disk until explicitly managed. The first backend recomputes the causal context per
decoded token rather than maintaining a KV cache. It prioritizes a verifiable inference path over throughput.

## Reproduce the effect

```bash
.venv-knn/bin/python scripts/demo_knn_lm.py
.venv-knn/bin/python -m pytest -q tests/test_knn_lm.py -p no:cacheprovider
```

The pretrained-model experiment on 14 November 2026 used the pinned tiny-GPT2 revision above, weight 0.75 and one neighbor:

| Datastore state | Next token | Its base probability | Its mixed probability |
|---|---|---|---|
| Empty | ` factors` | 0.002230% | 0.002230% |
| Checked continuation ` blue` | ` blue` | 0.002049% | 75.000512% |
| Same source replaced by ` green` | ` green` | 0.001988% | 75.000497% |
| Independent snapshot of the first revision | ` blue` | 0.002049% | 75.000512% |
| Source withdrawn | ` factors` | 0.002230% | 0.002230% |

The full weight hash before and after was identical. The test proves mutable token-level influence without parameter training;
the prompt was deliberately memorized. General improvement needs independent held-out contexts, target-token log loss/perplexity,
latency and adverse-case measurements. Select interpolation weight, neighbor count and temperature on validation data, then keep
them fixed for that evaluation. Retrieval exclusion by source is supported by the Python API for such measurements.

`vi-mcp knn evaluate --store DIR --model MODEL --revision COMMIT --data held-out.jsonl` computes teacher-forced base and kNN token
log loss and perplexity over reference rows with the same `{source, prompt, continuation}` structure. Each row's source is excluded
from retrieval. This prevents retrieving that exact source; choose the held-out partition before ingestion to exclude sibling
documents and paraphrases as well. Evaluation never writes reference targets into the datastore.
