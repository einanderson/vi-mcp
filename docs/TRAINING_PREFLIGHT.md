# Training-kit preflight

The VI produces learning data for a model that programs its machine. A successful export alone does not establish that a tokenizer,
chat template and trainer will preserve the complete programs. The kit now tests that path without training.

## Running the checks

Create a kit with `vi-mcp finetune kit DIR --base MODEL`, then work inside that private directory on the training host. Install
PyTorch appropriate to the host first, then `python -m pip install -r requirements-train.txt`. The generated requirements pin the
Transformers, TRL, PEFT, datasets and accelerate versions used in the integration test. GPU/QLoRA installation remains host-specific.

```bash
# No model weights or trainer loaded:
python train.py --stage logic --tokenizer-only --from-base
# Load the model, construct the actual trainer, collate and compute one loss:
python train.py --stage logic --preflight --from-base
# Check an existing preceding adapter instead:
python train.py --stage logic --preflight --from-adapter out/agentic
```

The stages are `sft`, `agentic`, `logic`, `kto` and `dpo`. Use `--cpu` where the model fits in RAM. Without a GPU, preflight selects
CPU automatically. `--revision COMMIT` pins a remote base/tokenizer. `--from-base` deliberately starts a fresh adapter directly from
the base for the selected stage; without it, later stages expect `out/sft` or an explicit `--from-adapter`. A preflight can probe a
missing parent with a fresh adapter, but reports `parent_ready=false`. Actual training refuses that missing dependency.
New adapters retain the resolved base revision. Later stages load that recorded revision and reject an explicitly different one.

`preflight-STAGE.json` records dataset, tokenizer, template and configuration hashes, dependency versions, token counts, rejected
classes, parent readiness and trainer verification. The full preflight also records one finite no-grad loss, batch shapes and an
unchanged parameter hash with zero optimizer steps. It calls neither `train()` nor `save_model()` and leaves `run.json` unchanged.
`--tokenizer-only` reports its narrower scope explicitly. A probe with another `--base` is marked as an override; it cannot certify
the later target model or accelerator. None of these reports declares the incubator mature or proves a learning gain.

## Complete programs and correct targets

The runtime reads original JSON before Arrow can merge distinct tool-argument dictionaries into columns with null fields. It
renders each complete assistant turn with the model's actual chat template, including tool declarations and calls. User text,
previous turns and tool observations remain context, with their loss labels masked. The prefix must match token for token; a
template that changes earlier tokens is refused. This also supports templates without assistant generation masks.

The default bound is the model's declared context window. `--sequence-length N` may lower it. An oversized conversation fails the
audit as a whole; `--skip-oversize` explicitly omits whole episodes and counts them. No answer or tool program is silently cut.
After constructing the trainer, its prepared token IDs and target masks must still match the audit. Preflight also verifies the
actual collated batch. Preference targets and KTO's mismatched KL completions must fit too. Machine receipts remain in their
checked program/preference formats, rather than becoming generic question-to-itself QA examples.

This is necessary because TRL's normal SFT default limits sequences to 1024 tokens, and its assistant-only option depends on
template generation masks. The kit uses audited pretokenized SFT labels and disables library truncation.
See the [official SFT trainer documentation](https://huggingface.co/docs/trl/main/en/sft_trainer),
[KTO trainer documentation](https://huggingface.co/docs/trl/main/en/kto_trainer) and
[DPO trainer documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer).

## Evaluation without the source checkout

The kit bundles the public `clingo_object` Python sources and the solver dependency, with source notice and hashes in `run.json`.
Package-local configuration and credentials are excluded. `program_evaluate.py` can therefore execute frozen cases without an
installed VI server. Cached predictions must cover exactly the held-out cases, including an explicitly empty cache.

Both generated evaluators use the shared provider compatibility helper. An incomplete model generation cannot receive a passing
grade even when its prefix happens to be a valid program. Ordinary independent judgments are bound to collected predictions and
the held-out file hash; changing the file invalidates that report. A successful preflight never substitutes for held-out evaluation.

## Evidence and limits

`tests/test_training_runtime.py` covers masking, distinct tool arguments, oversize refusal and changes introduced by a trainer.
`tests/test_training_runtime_neural.py` runs all five real TRL/PEFT trainers on an initialized tiny Transformer, forbidding training,
model saving and optimizer updates. It also tests a parent adapter and the no-weights tokenization path. These are API and format
checks, not training results. `tests/test_kit_evaluation.py` runs isolated generated scripts without the checkout and checks truncated
outputs and mismatched holdouts. The export privacy check is in `tests/test_anonym.py`.

The user has deferred actual fine-tuning. No adapter was trained and no production model weights were changed during this work.
The later large-model training run, hardware fit and before/after learning quality require their own measurements.
