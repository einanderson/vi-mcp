# Skill: fine-tune any open model with vi-mcp

Version 2026-11-16a. Follow the steps in order. Do one step at a time.

## Rules you must never break

1. Never start a training run without a clear "yes" from the person for THIS run.
2. Never stop or restart a service (for example a model server) without a clear "yes" from the person.
   If you stop one, write down what you stopped, when, and how to start it again (step 12 starts it again).
3. After every command, compare the output with the "OK if" line. If it does not match: stop, show the person the exact error text,
   and do not guess a fix.
4. Never put a token limit (`max_tokens`, `n_predict`) into a request to a language model. If an output is cut off, switch thinking off
   instead (`chat_template_kwargs: {"enable_thinking": false}`).
5. Never train on the evaluation data. Never change the evaluation files. The kit keeps them apart; do not move files between them.
6. Never copy the live database. The kit is built from a read-only copy (step 3 does this).
7. Write every result down: `vi_learn` with the numbers, the command and the host. A result that is not written down is lost.

Words used below:
- MODEL = the Hugging Face id of the base model, for example `Qwen/Qwen3-0.6B`.
- KIT = the kit directory, for example `/srv/ft/kit1`.
- GPU host = the computer with the graphics card that runs the training.

## Step 0 — Ask the person five questions

Ask all five in one message and wait for the answers:
1. Which base model? (MODEL)
2. Which GPU host, and which graphics card (name and memory in GB)?
3. Which services on the GPU host may be stopped during the run, if any?
4. How many hours may the run take?
5. What should the model learn: (a) the knowledge of this VI, or (b) a philosophy (name of the package, e.g. `kollektiv`)?

OK if: you have all five answers. If not: ask again for the missing ones. Do not continue without them.

## Step 1 — Check that the model can be trained

Run on any computer with Python and `transformers`:

```bash
python3 -c "from transformers import AutoTokenizer, AutoConfig; m='MODEL'; t=AutoTokenizer.from_pretrained(m); c=AutoConfig.from_pretrained(m); print('chat_template', bool(t.chat_template)); print('model_type', c.model_type); print('vocab', c.vocab_size)"
```

OK if: the output shows `chat_template True` and a `model_type`.
If `chat_template False`: STOP. Tell the person: "This model has no chat template; the kit cannot train it. Choose an instruct or chat
version of the model." If it fails with "trust_remote_code": STOP. Tell the person: "This model needs remote code; the kit does not run
remote code. Choose another model."
Do not suggest model names yourself — you may invent names that do not exist. Ask the person which model to use instead, then run step 1
again with it.

Write down the number of parameters. Find it in the model name (0.6B, 1.7B, 7B …) or on the model page.

## Step 2 — Choose the settings from the tables

Table A — precision and memory. Read the row of your graphics card memory and your model size.

| model size | GPU memory 8 GB | 16 GB | 24 GB | 48 GB or more / APU with shared memory |
|---|---|---|---|---|
| up to 1B | fp32, batch 1 | fp32, batch 1 | fp32, batch 1 | fp32, batch 1 |
| 1B – 3B | QLoRA (`--qlora`) | fp32 | fp32 | fp32 |
| 3B – 8B | not possible | QLoRA | QLoRA | bf16 (or fp32 on AMD) |
| 8B – 14B | not possible | not possible | QLoRA | QLoRA |

Notes for Table A:
- Measured: up to 1B in fp32 on 8 GB (RTX 3070) and 1.7B in fp32 on an AMD APU with 96 GB shared memory. The other cells are estimates;
  step 8 (preflight) tells you if the setting fits.
- On AMD Strix Halo (gfx1151) always add `--fp32` unless you use `--qlora`: bf16 has known errors there (ROCm/TheRock issue 9198).
- GRPO (step 9, philosophy only) needs much more memory than SFT. On 8 GB it ran out of memory even for a 0.6B model. Use 24 GB or more,
  or an APU with shared memory, for GRPO.
- The longest training example also costs memory: sequence length × vocab × 4 bytes for the logits alone. Example: 7,104 tokens ×
  151,936 vocab × 4 bytes ≈ 4.3 GB.

Table B — training settings. Use these unless the person says otherwise.

| setting | value | why |
|---|---|---|
| LoRA rank | 64 (`--lora-rank 64`) | rank 64 stored new facts; rank 16 did not (runs 2 and 3) |
| epochs | 2 for a philosophy, 4 for knowledge | measured in runs 3 and 15–19 |
| learning rate | the kit's default; philosophy stage B `--lr 2e-4` | runs 15–19 |
| model bigger than 1B | learning rate × 0.6 to × 0.95 | estimate (Thinking Machines, "LoRA Without Regret") |

OK if: you wrote down precision (fp32, bf16 or `--qlora`), rank, epochs and learning rate. If Table A says "not possible": STOP and tell
the person to choose a smaller model or a bigger GPU.

## Step 3 — Look at the plan of this VI

Call the tool: `vi_finetune(action="plan", base_model="MODEL")`.

OK if: the answer lists the steps and says whether the VI is mature (incubator). If it says "not mature": tell the person which criteria
are open. Continue only if the person says "yes, with --force".

## Step 4 — Write the kit

On the vi-mcp server. First make a read-only copy of the database, a config that points to the copy, and the evaluation questions next
to it (SQLite; for PostgreSQL use `pg_dump` instead of the first command):
```bash
mkdir -p /srv/ft/copy
python3 -c "import sqlite3; s=sqlite3.connect('file:/var/lib/vi-mcp/vi.db?mode=ro', uri=True); d=sqlite3.connect('/srv/ft/copy/vi.db'); s.backup(d); d.close()"
python3 -c "import json; c=json.load(open('/etc/vi-mcp/config.json')); c['db']='/srv/ft/copy/vi.db'; json.dump(c, open('/srv/ft/copy/config.json','w'))"
cp -r /var/lib/vi-mcp/eval /srv/ft/copy/
```
OK if: `/srv/ft/copy/vi.db` and `/srv/ft/copy/config.json` exist. (Use the paths of your server: `vi-mcp db status` shows the database.)

Then write the kit (`<copy>` below is `/srv/ft/copy`):

For knowledge (answer 5a):
```bash
VI_MCP_CONFIG=<copy>/config.json vi-mcp finetune kit KIT --base MODEL
```

For a philosophy (answer 5b), add `--philosophy NAME`:
```bash
VI_MCP_CONFIG=<copy>/config.json vi-mcp finetune kit KIT --base MODEL --philosophy NAME
```

Add `--force` only if the person said yes in step 3.

OK if: the directory KIT contains `run.json`, `train.py`, `training_runtime.py`, `requirements-train.txt` and a `split/` directory.
Open `KIT/run.json` and look at `split.stages`: every stage you want to train must have more than 0 train rows.
If a stage has 0 test rows, it cannot be measured: tell the person.

For knowledge only: if `documents.jsonl` is large, the kit's `synth_qa.py` writes question–answer pairs for it with a language model:
```bash
python synth_qa.py --url http://LLM-HOST/v1 --model NAME --limit 500
vi-mcp finetune split KIT
```
OK if: `synth_report.json` shows kept pairs. (Never use an object's own text as its question: the first run did that and learned to echo.)

## Step 5 — Copy the kit to the GPU host

```bash
rsync -a KIT/ GPU-HOST:/data/ft/kit1/
```

OK if: `ssh GPU-HOST ls /data/ft/kit1/run.json` prints the path.

## Step 6 — Prepare the container on the GPU host

Use a container. Do not install packages on the host itself.
- NVIDIA: a PyTorch CUDA image; run with `--gpus all`.
- AMD: a `rocm/pytorch` image whose tag lists your GPU (for gfx1151: ROCm 7.1 or newer); run with
  `--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined`.

Build an image that installs the kit's requirements without replacing the GPU version of torch:
```bash
pip freeze | grep -iE "^(torch|torchvision|torchaudio|pytorch-triton-rocm|triton)==" > keep.txt
pip install -c keep.txt -r requirements-train.txt
```
The image also needs `gcc` (the trainer compiles a small part).

Smoke test inside the container:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
OK if: it prints `True` and the name of the graphics card. If `False`: STOP and show the person the output.

Mount a Hugging Face cache into the container and download MODEL once. Then run training with `HF_HUB_OFFLINE=1`.

## Step 7 — Check the data without loading the model

Inside the container, in the kit directory:
```bash
python train.py --stage sft --tokenizer-only --fp32
```
For a philosophy kit use the first stage of the kit (`--stage language`) instead of `sft`.

OK if: the output contains `"valid": true` and `"rejected": {}`.
If rows are rejected because they are too long: do not cut them. Tell the person. They may allow `--skip-oversize` (whole rows are left out
and counted).

## Step 8 — Preflight: build the trainer, change nothing

```bash
python train.py --stage sft --preflight --fp32
```
(Use your precision from step 2 instead of `--fp32` if Table A said bf16 or `--qlora`.)

OK if: the output contains `"status": "preflight_passed"`. If it says "out of memory": go back to Table A and use the next smaller setting.

## Step 9 — Train (only after the person said yes)

Ask the person: "Preflight passed. Start the training now? It will take about H hours and stop SERVICE." Wait for "yes".

If the person allowed it, stop the service now and write down the time (rule 2).

Start the training detached, so it keeps running when you disconnect. Knowledge:
```bash
docker run -d --name ft-run1 <GPU flags> -v /data/ft/kit1:/kit -w /kit <image> python train.py --stage sft --fp32 --epochs 4 --lora-rank 64
```
Philosophy — the stages in this order, each one continues the one before:
```bash
python train.py --stage language --fp32 --epochs 2 --lora-rank 64
python train.py --stage philosophy --fp32 --epochs 2 --lr 2e-4
```
GRPO (optional, philosophy only, needs much memory, see Table A) comes after both: `python train.py --stage grpo --fp32`.

Watch the run every 30 minutes:
```bash
docker logs --tail 5 ft-run1
```
OK if: you see progress lines with a falling `loss`. STOP and tell the person if you see `Traceback`, `out of memory`, `Killed`, or a loss of
`nan`.

The run is done when the container has exited. Check its exit code: `docker inspect ft-run1 --format '{{.State.ExitCode}}'`.
OK if: `0`.

## Step 10 — Measure

Measure only with held-out data. Never with the training loss.

Knowledge (answer 5a): the knowledge probe asks new questions about trained documents.
```bash
python probe.py answer
python probe.py judge
python probe.py summarize
```
OK if: `probe/probe_eval.json` exists. Report: correct / partial / wrong for base and tuned model, and the sign-flip test `p` value.

Philosophy (answer 5b): the solver scores every answer; no judge model.
```bash
python phil_answers.py --adapter out/philosophy --fp32
python phil_score.py
```
OK if: `eval/philosophy_score.json` exists and every set shows `items` and `world_exact`. Report for each set: tuned `world_exact`, and
`invented_certainty` (it must stay near 0). If many answers are marked `budget` (cut by the generation limit of the local script), run
`phil_answers.py` again with `--budget 1536`. The prose items are scored by the round trip: `python phil_roundtrip.py` (see its help).

For a fair comparison with an earlier run, use the same evaluation items and count the items that only one of the two runs got right
(a paired test). A difference of 2–3 items out of 40 is noise.

## Step 11 — Record the run

Call the tool: `vi_finetune(action="record", directory="KIT")` (the kit directory on the vi-mcp server; copy `out/` back first).

OK if: the answer names a model object with the dataset hash, the base model and the adapter hash.
Then write the result down with `vi_learn` (rule 7): what was trained, the numbers of step 10, the host and the date.

## Step 12 — Clean up

1. Start every service you stopped (rule 2). Then test it with a small request (for example "Say only OK.") and check the answer — a
   health check alone is not enough.
2. Tell the person: the result (step 10), where the adapter is (`out/`), and that it is not deployed. Deploy a tuned model only after
   the person says yes.

OK if: every service you stopped answers the small request again, and the person has the result. If a service does not answer: tell the
person at once, with the command you used to start it and its output.

## What made a philosophy model better (19 measured runs, docs/FINETUNING_PAPER.md)

Use these when you build training data for a philosophy yourself. Each line is a measured result:
1. Answer format first lists the facts, then judges (exact verdicts 18 → 29 of 40).
2. Enumerate: one line per item with its reason, then the totals (29 → 39 of 40).
3. Number the input sentences and write one output line per sentence, with "unknown" for a sentence that states nothing (held-out
   phrasings 24 → 37 of 40; fact lists 11 → 40 of 40).
4. Put the rule set itself into the training targets if the model must recite it, in a fixed order with an anchor per rule (P5.1: …).
5. Keep the language stage in the seed stage (1,000 replayed rows), or the model forgets it.
6. The language stage must contain every rule form the seed uses (constants in heads, any variable order, several rules on one body).
7. GRPO only sharpens what SFT already does; check pass@k before you spend GPU hours on it.

## When something goes wrong

| what you see | cause | what to do |
|---|---|---|
| `out of memory` | model or example too big | Table A: next smaller setting; or a host with more memory |
| loss is `nan` | bf16 on a GPU with bf16 errors | add `--fp32` |
| `chat_template_error` in step 7 | rows the template cannot render | tell the person; do not edit rows by hand |
| `permission denied` on `~/.docker/buildx` | Docker config of another user | set `DOCKER_CONFIG` to your own directory; do not change the user's files |
| files in the kit belong to root | the container writes as root | read them with `sudo`, or run the container with `-u $(id -u)` |
| the model repeats the question or a document | trained on an object's own text as the question | rebuild the kit (step 4); use `synth_qa.py` pairs |
| the tuned model invents hosts or facts | expected after knowledge fine-tuning | report it; truth stays with the VI and its seed, not the model |
