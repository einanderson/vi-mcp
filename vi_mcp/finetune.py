"""Fine-tuning an AI with what a VI learned (0.62.0) — the VI knows how, and an AI on vi-mcp is told "I want to fine-tune".

The user (13.11.2026): "Die VI ist noch nicht fertig gebrütet, aber die VI muss wissen, wie man fine-tuned, und wenn man den
vi-mcp-Server benutzt, soll man dem LLM einfach sagen, dass man fine-tunen will." and "Es soll eine direkte Schnittstelle für das
Fine-Tuning geben."

vi-mcp trains no model by itself. This module is the interface a person uses through their AI (tool `vi_finetune`) or the command line:
  guide    the handbook below — what fine-tuning is, which program, which hardware, how back into llama.cpp, how to measure
  plan     this VI's readiness (incubator) and the concrete steps for this server
  kit      a ready directory: the training data (incubator hatch), a test split, train.py (TRL + LoRA), convert.sh (GGUF adapter),
           evaluate.py (base against fine-tuned on held-out questions), README — written, never run by vi-mcp
  record   after a run: a model object with its lineage — dataset manifest, base model, adapter hash, evaluation, incubator state
Research behind the handbook (13.11.2026, primary sources named in it).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from . import __version__
from . import memory as g

GUIDE_VERSION = "2026-11-14"
TEST_SHARE = 0.15
STRATIFY_MIN = 5  # a stage with at least this many groups always gets held-out groups

GUIDE = """# Fine-tuning an AI with the knowledge of a VI

## What it is
An existing open model (e.g. Qwen, Llama) stays frozen; only a small add-on is trained — a **LoRA adapter** (a few percent of the
model's size). **QLoRA** loads the frozen model in 4 bit so it fits into little GPU memory (Dettmers et al., arXiv 2305.14314). The data
stays in the house; only the base model is downloaded once.

## Training formats of `vi-mcp dataset export`
- **SFT** — `qa.jsonl`: a real question (doc2query) → knowledge with source; imported examples are not automatically verified
  answers, the manifest reports their evidence and question source. Knowledge without a real question is in `documents.jsonl`
  and is never trained as its own question: the first production run (14.11.2026) did that for 96 % of its rows and learned to
  echo and continue documents (no gain on real questions, loops). The kit's `synth_qa.py` turns documents into grounded
  question-answer pairs (`qa_synth.jsonl`); diverse QA pairs inject knowledge better than raw text (arXiv 2409.07431, 2412.14964).
- **KTO** — `kto.jsonl`: an answer was good or not (binary). Exactly the VI's helped/not-helped signal (Ethayarajh et al., arXiv 2402.01306).
- **DPO** — `dpo.jsonl`: in the same situation this answer was better than that one (Rafailov et al., arXiv 2305.18290).
- **agentic** — `agentic.jsonl`: the model asks the VI with `vi_recall` in VIL and answers from what it found (tool-use SFT).
- **logic** — `logic.jsonl`: the model writes clingo-object programs for the tasks of saved rule sets and answers from the solver
  (tool-use SFT with `vi_logic`, 0.63.0 — "clingo ist die Sprache für LLMs"); wrongly rated conclusions are KTO negatives.

## The program
- **TRL** (Hugging Face) supports SFT `messages`, KTO `prompt/completion/label`, DPO `prompt/chosen/rejected`
  (https://huggingface.co/docs/trl/main/en/dataset_formats). KTO: learning rate 5e-7 … 5e-6, desirable:undesirable between 1:1 and 4:3
  (https://huggingface.co/docs/trl/main/en/kto_trainer).
- **Unsloth** builds on TRL: "2× faster with 70% less VRAM", supports AMD/ROCm incl. gfx1151 (https://github.com/unslothai/unsloth,
  https://unsloth.ai/docs/basics/amd). **LLaMA-Factory** has a web UI (KTO needs the column `label` mapped to `kto_tag`).
  **torchtune** is no longer maintained.

## Hardware (minimum per Unsloth, https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements)
| model | QLoRA 4 bit | LoRA 16 bit |
|---|---|---|
| 3B | 3.5 GB | 8 GB |
| 7–8B | 5–6 GB | 19–22 GB |
| 14B | 8.5 GB | 33 GB |
An 8 GB card: QLoRA up to ~8B with short sequences. A 96 GB unified-memory APU: LoRA in 16 bit up to 14B — test first whether PyTorch
sees the GPU (`torch.cuda.is_available()` on ROCm).

## Back into llama.cpp
Keep the adapter separate: `python convert_lora_to_gguf.py ADAPTER_DIR --base HF_BASE_DIR --outtype f16`, then
`llama-server -m base.gguf --lora adapter.gguf` (switchable at runtime with `--lora-init-without-apply` and `POST /lora-adapters`)
(https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). Or merge in 16 bit, `convert_hf_to_gguf.py`, `llama-quantize`.

## The steps
1. The VI is mature enough: `vi-mcp incubator status` (a run before is a trial).
2. `vi-mcp finetune kit DIR` — data, test split, scripts. Then `python synth_qa.py --url http://HOST/v1 --model NAME [--limit N]`
   (an instruction model writes up to 3 grounded pairs per document; resumable; report in synth_report.json) and
   `vi-mcp finetune split DIR` so the pairs join the shared split. Check `run.json` → split → qa_question_sources.
3. On the training host install its hardware-appropriate PyTorch, then `python -m pip install -r requirements-train.txt`.
   First run `python train.py --stage logic --tokenizer-only --from-base` (no model weights), then
   `python train.py --stage logic --preflight --from-base` (real trainer, one no-grad loss, zero optimizer steps).
   Repeat for the intended stages. `--cpu` is available; the model must fit in memory. `--fp32` keeps float32 on a GPU (no bf16 mixed precision),
   e.g. on AMD gfx1151 where bf16 problems are reported. Reports are `preflight-STAGE.json`.
   These checks do not train, save an adapter, change run.json or certify incubator maturity. A missing parent adapter is named in the report.
4. When explicitly ready, train SFT (`python train.py --stage sft`), then optionally agentic or logic. Later stages read only their shared train split.
   Use `--stage logic --from-adapter out/agentic` to continue the agentic stage, or `--stage dpo --from-adapter out/logic`
   to continue the logic stage; without this flag each later stage starts from out/sft. `--stage logic --from-base` explicitly starts
   program training directly on a fresh adapter without a prior QA stage. Solver-only examples and
   observational hint pairs are withheld from default program/DPO training. Checked machine examples are published explicitly with vi_machine teach.
5. `convert.sh` → GGUF adapter; start llama-server with base and adapter.
6. `evaluate.py` collects paired predictions; supply independent --judgments for correctness. For program writing use
   `program_evaluate.py --base URL --tuned URL`: full solver answers on frozen held-out worlds, not word overlap.
   The kit bundles the solver's Python sources, so it works without a VI checkout. Truncated model outputs cannot count as correct.
   `heldout_loss.py --stage sft|agentic|logic` measures the held-out loss on the assistant targets, for base and adapter, with the
   same tokens as training. It shows knowledge absorption, not task success, and it is never the adoption gate.
   Adoption requires an exact one-sided sign-flip test over independent task families (clustered McNemar) at p < 0.05 with a
   positive difference. It assumes independent families and, under the null, base and tuned exchangeable within each family.
   The Hoeffding lower_95 is reported as the worst-case size bound.
7. `vi-mcp finetune record DIR` — the run becomes a model object with its lineage in the VI.

## Rules
- Never set a token cap (max_tokens) in evaluation; if answers run long, switch thinking off (`enable_thinking: false`).
- Evaluation source pages, duplicate contents and questions are excluded from export. The kit shares one split across all stages; never train from the unsplit dataset files when measuring the held-out sets.
- The kit renders tool calls before Arrow loading and learns only complete assistant targets, masking user text and tool observations.
  Every original token and mask is checked against the real trainer. No implicit 1024-token truncation is accepted: examples must fit
  the model's declared context (`--sequence-length` may set a smaller bound). Oversized episodes refuse preparation; `--skip-oversize`
  explicitly omits whole episodes with counts and reasons. Invalid templates or changed trainer tokens always refuse preparation.
- Pin a remote model/tokenizer with `--revision COMMIT`. Preflight with `--base` may probe a small fixture; the report marks the override.
  Such a probe verifies that fixture, not the intended large model, GPU or subsequent learning quality.
- Internal data: the files carry company knowledge unanonymised.
"""


def _cfg() -> dict:
    from . import config
    try:
        return config.read_raw(config.fpath()).get("finetune") or {}
    except Exception:  # noqa: BLE001
        return {}


def plan(index: str | None = None, base_model: str | None = None) -> dict:
    """Readiness and the concrete steps for this server."""
    from . import incubator
    st = incubator.status(index)
    base = base_model or _cfg().get("base_model")
    steps = [
        ("readiness", "vi-mcp incubator status" + (f" --index {index}" if index else ""),
         "mature" if st["zustand"] == "reif" else f"not mature — a run now is a trial; open: {', '.join(st['offen'])}"),
        ("kit", f"vi-mcp finetune kit /srv/vi-finetune{(' --index ' + index) if index else ''}{(' --base ' + base) if base else ''}"
                + ("" if st["zustand"] == "reif" else " --force"), "training data, test split and scripts, nothing is run"),
        ("preflight", "python train.py --stage logic --tokenizer-only --from-base; python train.py --stage logic --preflight --from-base",
         "install hardware-appropriate torch and requirements-train.txt first; real trainer check, no training"),
        ("train", "python train.py --stage sft   (then optionally --stage logic, --stage kto or --stage dpo)", "only on an explicit training decision; a preflight alone changes no weights"),
        ("convert", "./convert.sh", "the adapter as GGUF for llama.cpp, and the llama-server line"),
        ("evaluate", "python evaluate.py --base URL --tuned URL", "same held-out questions to both; adopt only if better"),
        ("record", "vi-mcp finetune record /srv/vi-finetune", "the run as a model object with its lineage"),
    ]
    missing = [] if base else ["base_model (config finetune.base_model or --base, e.g. a Hugging Face id of the model already served)"]
    return {"zustand": st["zustand"], "offen": st["offen"], "grundmodell": base, "fehlt": missing, "schritte": steps, "handbuch": GUIDE_VERSION}


def plan_lines(p: dict) -> list[str]:
    head = ("This VI is mature: its training data can fine-tune an AI." if p["zustand"] == "reif"
            else f"This VI is still in the incubator (open: {', '.join(p['offen'])}) — a fine-tuning now is a trial run.")
    out = [head, f"Base model: {p['grundmodell'] or '— not set'}" + (f" · needed: {'; '.join(p['fehlt'])}" if p["fehlt"] else ""), "Steps:"]
    out += [f"  {i}. {name}: {cmd}  — {why}" for i, (name, cmd, why) in enumerate(p["schritte"], 1)]
    out.append("vi_finetune(action='guide') explains fine-tuning; the kit is written on the server by a person (vi-mcp finetune kit).")
    return out


TRAIN_PY = r'''"""Fine-tuning kit entry point. First run --preflight (no training); --tokenizer-only loads no model weights.
Stages: sft, agentic, logic, kto, dpo. Use --from-adapter to continue a selected prior stage.
"""
from training_runtime import main

if __name__ == "__main__":
    raise SystemExit(main())
'''


EVALUATE_PY = r'''"""Collect paired predictions; correctness comes from independent judgments, never keyword overlap."""
import argparse, hashlib, json
from clingo_object.agent import chat_post
from evaluation_core import summarize_judgments

a = argparse.ArgumentParser()
a.add_argument("--base", required=True)
a.add_argument("--tuned", required=True)
a.add_argument("--judgments", help="JSON list: id, group, base_correct, tuned_correct, reference")
o = a.parse_args()
rows = [json.loads(l) for l in open("split/test.jsonl") if l.strip()]
evaluation_hash = hashlib.sha256(open("split/test.jsonl", "rb").read()).hexdigest()
if o.judgments:
    prediction_file = json.load(open("predictions.json"))
    if prediction_file.get("evaluation_set_sha256") != evaluation_hash:
        raise ValueError("predictions do not belong to the held-out file")
    predictions = {p["id"]: p for p in prediction_file["predictions"]}

    judgments = json.load(open(o.judgments))
    ids = {r["meta"]["id"] + ":" + str(i) for i, r in enumerate(rows)}
    if {r["id"] for r in judgments} != ids or len(judgments) != len(rows):
        raise ValueError("judgments must cover exactly the held-out questions")
    for i, r in enumerate(rows):
        j = next(j for j in judgments if j["id"] == r["meta"]["id"] + ":" + str(i))
        j["group"] = r["meta"].get("group", r["meta"]["id"])
    if set(predictions) != ids or len(prediction_file["predictions"]) != len(rows):
        raise ValueError("predictions must cover exactly the held-out questions")
    incomplete = {"base": 0, "tuned": 0}
    for judgment in judgments:
        prediction = predictions[judgment["id"]]
        for name in ("base", "tuned"):
            if prediction.get(name + "_complete") is not True:
                judgment[name + "_correct"] = False
                incomplete[name] += 1
    summary = summarize_judgments(judgments)
    summary["incomplete_predictions"] = incomplete
    summary["predictions_sha256"] = hashlib.sha256(open("predictions.json", "rb").read()).hexdigest()
    summary["evaluation_set_sha256"] = evaluation_hash
    json.dump(summary, open("eval.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
else:
    predictions = []
    for i, r in enumerate(rows):
        q = r["messages"][0]["content"]
        item = {"id": r["meta"]["id"] + ":" + str(i), "question": q, "reference_answer": r["messages"][1]["content"]}
        for name, url in (("base", o.base), ("tuned", o.tuned)):
            body = {"messages": [{"role": "user", "content": q}], "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            choice = chat_post(url, body, timeout=600)["choices"][0]
            item[name] = choice["message"].get("content") or ""
            item[name + "_complete"] = choice.get("finish_reason") == "stop"
        predictions.append(item)
    json.dump({"evaluation_set_sha256": evaluation_hash, "predictions": predictions}, open("predictions.json", "w"), indent=2)
    print("Predictions collected. Supply independent --judgments; no correctness score has been inferred.")
'''


SYNTH_QA_PY = r'''"""Question-answer pairs for documents that have no real question yet (dataset/documents.jsonl -> dataset/qa_synth.jsonl).

Why (14.11.2026): training "question = the document's own text, answer = the document" taught a model to echo and continue documents
and to loop; it answered no real question better. Knowledge injection works better from diverse question-answer pairs than from raw
text (Yang et al., Synthetic continued pretraining, arXiv 2409.07431; Kujanpaeae et al., arXiv 2412.14964).
A language model writes up to 3 pairs per document through an OpenAI-compatible API. Each pair is kept only if it is grounded: most
answer words occur in the document, the question names its subject instead of "the document", nothing is copied from the document's
beginning. Rejections are counted per reason. Resumable: documents already in qa_synth.jsonl or synth_rejected.jsonl are skipped.
No token cap (the model ends its own answer); thinking is switched off. Then run `vi-mcp finetune split KIT` (or rerun the kit) so the
pairs join the shared split. The pairs are model output checked for word overlap, not verified by a person.
"""
import argparse, concurrent.futures as cf, hashlib, json, random, re, threading, time, urllib.request

PAIRS = 3
SYSTEM = ("You write training questions for a company knowledge base. You get one document. Write up to 3 questions that an employee "
          "would really ask and that this document answers, each with a short answer of one to three sentences. Vary the facts and the "
          "wording; when asked for more than 3, ask about the same facts in different words too. Use only facts stated "
          "in the document; copy names, paths, hosts, commands and numbers exactly. Each question must make sense without seeing the "
          "document: name the system, host or topic, never write 'the document', 'this page' or 'the text'. Write in the language of "
          "the document. If the document states no answerable fact (an empty table, only a title), return an empty list. "
          'Reply with JSON only: {"pairs": [{"question": "...", "answer": "..."}]}')
PROMPT_SHA256 = hashlib.sha256(SYSTEM.encode()).hexdigest()
DOC_CHARS = 6000
GROUNDED = 0.8
META = re.compile(r"\b(the|this|that) (document|page|text|article)\b|\b(dieses?|das|der|die|im) (dokument|seite|text|artikel)\b", re.I)
WORD = re.compile(r"[\w./:-]{3,}")


def words(text):
    return {w.lower().strip(".:/-") for w in WORD.findall(text)} - {""}


def check(pair, document):
    q, a = (pair.get("question") or "").strip(), (pair.get("answer") or "").strip()
    if not 15 <= len(q) <= 300 or not 2 <= len(a) <= 800:
        return "length"
    if META.search(q):
        return "refers to the document instead of its subject"
    head = re.sub(r"\s+", " ", document[:200]).lower()
    if re.sub(r"\s+", " ", q).lower()[:60] in head:
        return "copied from the document's beginning"
    aw, dw = words(a), words(document)
    if not aw or len(aw & dw) / len(aw) < GROUNDED:
        return "not grounded in the document"
    return None


def system_for(pairs):
    return SYSTEM.replace("Write up to 3 questions", f"Write up to {pairs} questions")


def jaccard(a, b):
    x, y = words(a), words(b)
    return len(x & y) / max(1, len(x | y))


def ask(url, model, document, system=SYSTEM):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": document[:DOC_CHARS]}],
            "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        choice = json.load(r)["choices"][0]
    text = choice["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))["pairs"] if m else None, choice.get("finish_reason")


def main():
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8080/v1")
    a.add_argument("--model", required=True)
    a.add_argument("--limit", type=int, default=0, help="at most this many documents in this run (seeded random order); 0 = all")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--seed", type=int, default=1114)
    a.add_argument("--pairs", type=int, default=PAIRS, help="pairs asked per document; more varied pairs per fact inject knowledge "
                   "better (paraphrase scaling: arXiv 2312.05934, 2411.00686)")
    a.add_argument("--more", help="file of object ids (one per line): ask these documents again for further pairs, even if done")
    a.add_argument("--hold-out", help="JSONL of {id, question} kept out of training (evaluation probes): a pair whose question is "
                   "this close (word Jaccard >= 0.5) to one of them is rejected")
    o = a.parse_args()
    system = system_for(o.pairs)
    prompt_sha256 = hashlib.sha256(system.encode()).hexdigest()
    held = {}
    if o.hold_out:
        for l in open(o.hold_out, encoding="utf-8"):
            if l.strip():
                h = json.loads(l)
                held.setdefault(h["id"], []).append(h["question"])
    asked = {}
    try:
        for l in open("dataset/qa_synth.jsonl", encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                asked.setdefault(r["meta"]["id"], set()).add(" ".join(sorted(words(r["messages"][0]["content"]))))
    except FileNotFoundError:
        pass
    docs = [json.loads(l) for l in open("dataset/documents.jsonl", encoding="utf-8") if l.strip()]
    done = set()
    for name in ("dataset/qa_synth.jsonl", "dataset/synth_rejected.jsonl"):
        try:
            done |= {json.loads(l)["meta"]["id"] for l in open(name, encoding="utf-8") if l.strip()}
        except FileNotFoundError:
            pass
    again = {l.strip() for l in open(o.more, encoding="utf-8") if l.strip()} if o.more else set()
    todo = [d for d in docs if d["meta"]["id"] in again] if o.more else [d for d in docs if d["meta"]["id"] not in done]
    random.Random(o.seed).shuffle(todo)
    todo = todo[: o.limit] if o.limit else todo
    lock, stats, t0 = threading.Lock(), {"documents": 0, "pairs": 0, "rejected": {}}, time.time()
    out = open("dataset/qa_synth.jsonl", "a", encoding="utf-8")
    rej = open("dataset/synth_rejected.jsonl", "a", encoding="utf-8")

    def one(d):
        try:
            pairs, finish = ask(o.url, o.model, d["document"], system)
        except Exception as e:  # noqa: BLE001 — a failed call is counted, never a silent gap; the next run retries it
            return d, None, f"request failed: {type(e).__name__}"
        if pairs is None or finish != "stop":
            return d, [], "no JSON answer" if finish == "stop" else f"finish_reason {finish}"
        return d, pairs, None

    with cf.ThreadPoolExecutor(o.workers) as pool:
        for d, pairs, error in pool.map(one, todo):
            with lock:
                stats["documents"] += 1
                kept, reasons = [], []
                for p in pairs or []:
                    why = check(p, d["document"]) if isinstance(p, dict) else "format"
                    if not why and any(jaccard(p["question"], h) >= 0.5 for h in held.get(d["meta"]["id"], [])):
                        why = "too close to a held-out evaluation question"
                    key = " ".join(sorted(words(p["question"]))) if isinstance(p, dict) else ""
                    if not why and key in asked.setdefault(d["meta"]["id"], set()):
                        why = "duplicate question"
                    if why:
                        reasons.append(why)
                    else:
                        asked[d["meta"]["id"]].add(key)
                        kept.append(p)
                if error:
                    reasons.append(error)
                for why in reasons:
                    stats["rejected"][why] = stats["rejected"].get(why, 0) + 1
                meta = {**d["meta"], "question_source": "synthetic", "generator": o.model, "prompt_sha256": prompt_sha256}
                for p in kept:
                    answer = p["answer"].strip() + f"\n\nSource: {d['source']}"
                    out.write(json.dumps({"messages": [{"role": "user", "content": p["question"].strip()},
                                                       {"role": "assistant", "content": answer}], "meta": meta}, ensure_ascii=False) + "\n")
                stats["pairs"] += len(kept)
                if not kept and not (error or "").startswith("request failed"):
                    rej.write(json.dumps({"meta": {"id": d["meta"]["id"]}, "reasons": reasons}, ensure_ascii=False) + "\n")
                out.flush(); rej.flush()
                if stats["documents"] % 20 == 0:
                    print(f"{stats['documents']}/{len(todo)} documents, {stats['pairs']} pairs, {round(time.time() - t0)} s", flush=True)
    stats.update(seconds=round(time.time() - t0), generator=o.model, prompt_sha256=prompt_sha256, pairs_asked=o.pairs,
                 remaining=len([d for d in docs if d["meta"]["id"] not in done]) - (0 if o.more else len(todo)))
    json.dump(stats, open("synth_report.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
'''


CONVERT_SH = """#!/bin/sh
# The adapter as GGUF for llama.cpp — written by vi-mcp finetune kit. LLAMA_CPP = a checkout of llama.cpp, BASE_HF = the base model's
# Hugging Face directory (the same one train.py used).
set -e
: "${LLAMA_CPP:?set LLAMA_CPP to your llama.cpp checkout}"
: "${BASE_HF:?set BASE_HF to the base model directory}"
STAGE="${1:-sft}"
python "$LLAMA_CPP/convert_lora_to_gguf.py" "out/$STAGE" --base "$BASE_HF" --outtype f16 --outfile "adapter-$STAGE.gguf"
echo "serve: llama-server -m BASE.gguf --lora adapter-$STAGE.gguf   (switchable: --lora-init-without-apply, POST /lora-adapters)"
"""


def _split(qa_path: Path, out_dir: Path, share: float = TEST_SHARE, seed: int = 7) -> dict:
    """One connected-component split across ALL stages: object, source, duplicate content and program family stay together."""
    data = {name: [json.loads(s) for s in (qa_path.parent / f"{name}.jsonl").read_text().splitlines() if s.strip()]
            if (qa_path.parent / f"{name}.jsonl").exists() else [] for name in ("qa", "agentic", "logic", "kto", "dpo")}
    synth = qa_path.parent / "qa_synth.jsonl"  # synth_qa.py output: QA rows like qa.jsonl, split with their source object's group
    if synth.exists():
        data["qa"] += [json.loads(s) for s in synth.read_text().splitlines() if s.strip()]
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]
    def keys(row):
        m = row.get("meta") or {}
        refs = ["id:" + str(m[k]) for k in ("id", "chosen", "rejected") if m.get(k)]
        refs += [k + ":" + str(m[k]) for k in ("group", "content_hash", "program_family") if m.get(k)]
        return refs or ["row:" + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()]
    for rows in data.values():
        for r in rows:
            refs = keys(r)
            for x in refs[1:]:
                parent[root(x)] = root(refs[0])
    groups = sorted({root(keys(r)[0]) for rows in data.values() for r in rows})
    rnd = random.Random(seed)
    test_groups = set(rnd.sample(groups, max(1, int(len(groups) * share)))) if len(groups) > 1 else set()
    # Stratified per stage (14.11.2026): the joint sample above can miss a small stage entirely — the production kit's logic stage
    # had 14 groups and 0 test rows, so its gain was unmeasurable. Each stage with at least STRATIFY_MIN groups gets its share of
    # test groups; components stay joint across stages, so this adds held-out groups and never splits one.
    for rows in data.values():
        own = sorted({root(keys(r)[0]) for r in rows})
        if len(own) >= STRATIFY_MIN:
            want = max(1, round(len(own) * share))
            have = [x for x in own if x in test_groups]
            if len(have) < want:
                test_groups |= set(rnd.sample([x for x in own if x not in test_groups], want - len(have)))
    out_dir.mkdir(exist_ok=True)
    counts = {}
    for stage, rows in data.items():
        train = [r for r in rows if root(keys(r)[0]) not in test_groups
                 and not (stage in ("logic", "dpo") and (r.get("meta") or {}).get("validation") in ("solver_only", "observational"))]
        test = [r for r in rows if root(keys(r)[0]) in test_groups]
        for side, part in (("train", train), ("test", test)):
            path = out_dir / (f"{side}.jsonl" if stage == "qa" else f"{stage}.{side}.jsonl")
            _write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
        counts[stage] = {"train": len(train), "test": len(test), "withheld": len(rows) - len(train) - len(test),
                         "groups": len({root(keys(r)[0]) for r in rows}), "test_groups": len({root(keys(r)[0]) for r in test})}
        if counts[stage]["withheld"]:
            counts[stage]["reason"] = "solver-only program demonstration or observational preference; no independent task verification"
    test_ids = {r["meta"]["id"] for r in data["qa"] if root(keys(r)[0]) in test_groups}
    sources = {side: dict(Counter((r.get("meta") or {}).get("question_source") for r in data["qa"] if (root(keys(r)[0]) in test_groups) == (side == "test")))
               for side in ("train", "test")}
    return {**counts["qa"], "objekte_test": len(test_ids), "groups": len(groups), "test_groups": len(test_groups), "stages": counts,
            "qa_question_sources": sources}


def resplit(directory: str) -> dict:
    """Redo the shared split after synth_qa.py added dataset/qa_synth.jsonl; run.json records the new split and the synth file's hash."""
    d = Path(directory)
    if not (d / "run.json").exists():
        raise ValueError(f"{d}: no run.json — not a kit written by vi-mcp finetune kit")
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    run["split"] = _split(d / "dataset" / "qa.jsonl", d / "split")
    synth = d / "dataset" / "qa_synth.jsonl"
    run["qa_synth_sha256"] = hashlib.sha256(synth.read_bytes()).hexdigest() if synth.exists() else None
    _write(d / "run.json", json.dumps(run, indent=2, ensure_ascii=False))
    return run["split"]


HELDOUT_LOSS_PY = r'''"""Held-out next-token loss of the base model and, optionally, a LoRA adapter on one stage's TEST split.

Same tokens and masks as training (training_runtime.prepare: assistant targets only), a seeded sample of whole held-out rows.
Lower loss on held-out rows shows that the model absorbed the VI's knowledge for that stage. It is NOT the adoption gate:
task success needs independent judgments (evaluate.py) or executable references (program_evaluate.py).
    python heldout_loss.py --stage sft [--adapter out/sft] [--base-only] [--examples 400] [--fp32]
"""
import argparse, json, math, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from training_runtime import prepare

FILES = {"sft": "split/test.jsonl", "agentic": "split/agentic.test.jsonl", "logic": "split/logic.test.jsonl"}
a = argparse.ArgumentParser()
a.add_argument("--stage", choices=sorted(FILES), default="sft")
a.add_argument("--adapter", help="LoRA adapter directory (default out/<stage>)")
a.add_argument("--base-only", action="store_true")
a.add_argument("--examples", type=int, default=400)
a.add_argument("--fp32", action="store_true", help="float32 weights (default on CPU)")
a.add_argument("--sequence-length", type=int, default=4096)
o = a.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if (o.fp32 or device == "cpu") else "auto"
base = json.load(open("run.json"))["base_model"]
rows = [json.loads(l) for l in open(FILES[o.stage]) if l.strip()]
if not rows:
    raise SystemExit(f"{FILES[o.stage]} is empty: no held-out rows for stage {o.stage}")
random.Random(7).shuffle(rows)
tok = AutoTokenizer.from_pretrained(base)
data, audit = prepare(rows, tok, o.stage, o.sequence_length, skip_oversize=True)
data = data[:o.examples]


def score(model):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ex in data:
            ids = torch.tensor([ex["input_ids"]], device=device)
            labels = torch.tensor([ex["labels"]], device=device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            target = labels[0, 1:]
            mask = target != -100
            total -= float(torch.log_softmax(logits[mask], dim=-1).gather(1, target[mask].unsqueeze(1)).sum())
            count += int(mask.sum())
    return total / count, count


t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
base_nll, tokens = score(model)
out = {"metric": "held_out_assistant_token_nll", "stage": o.stage, "file": FILES[o.stage], "examples": len(data), "tokens": tokens,
       "sample_seed": 7, "device": device, "dtype": str(dtype), "base_nll": base_nll, "base_perplexity": math.exp(min(700, base_nll)),
       "oversize_skipped": audit["rejected"].get("oversize", 0),
       "scope": "held-out rows of this kit's split; knowledge absorption, not verified task success"}
if not o.base_only:
    from peft import PeftModel
    adapter = o.adapter or f"out/{o.stage}"
    tuned_nll, _ = score(PeftModel.from_pretrained(model, adapter))
    out.update(adapter=adapter, tuned_nll=tuned_nll, difference=tuned_nll - base_nll, tuned_perplexity=math.exp(min(700, tuned_nll)))
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(f"heldout-loss-{o.stage}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
'''


PROGRAM_EVALUATE_PY = r'''"""Evaluate program generation on frozen held-out tasks. No training or live tools are executed."""
import argparse, hashlib, json
from clingo_object.agent import program_of, SYSTEM, chat_post
from evaluation_core import evaluate_program, summarize_judgments

a = argparse.ArgumentParser()
a.add_argument("--base", required=True, help="base chat completions URL")
a.add_argument("--tuned", required=True, help="tuned chat completions URL")
a.add_argument("--predictions", help="optional JSON list: id, base program, tuned program (offline replay)")
o = a.parse_args()
cases = [json.loads(l) for l in open("program_eval.jsonl") if l.strip()]
cached_rows = json.load(open(o.predictions)) if o.predictions else []
cached = {r["id"]: r for r in cached_rows}
if o.predictions and (set(cached) != {r["id"] for r in cases} or len(cached_rows) != len(cases)):
    raise ValueError("cached predictions must cover exactly the held-out tasks")
judgments, details = [], []
for row in cases:
    outputs = {}
    for name, url in (("base", o.base), ("tuned", o.tuned)):
        if o.predictions:
            source = cached[row["id"]][name]
        else:
            q = row["question"] + "\nFrozen objects (use only these tags and filters):\n" + json.dumps(row["case"].get("input_objects", row["case"].get("objects", {})))
            body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            choice = chat_post(url, body, timeout=600)["choices"][0]
            if choice.get("finish_reason") != "stop":
                outputs[name] = {"correct": False, "error": "incomplete_model_output", "state": "refused"}
                continue
            source = program_of(choice["message"].get("content") or "")
        outputs[name] = evaluate_program(source, row["case"])
    judgments.append({"id": row["id"], "group": row["group"], "base_correct": outputs["base"]["correct"],
                      "tuned_correct": outputs["tuned"]["correct"], "reference": row["reference"]})
    details.append({"id": row["id"], **outputs})
summary = summarize_judgments(judgments)
summary["evaluation_set_sha256"] = hashlib.sha256(open("program_eval.jsonl", "rb").read()).hexdigest()
summary["evaluation_kind"] = "program"
json.dump({**summary, "details": details}, open("program_eval_result.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
'''


def _write(p: Path, text: str, mode: int = 0o600) -> None:
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(p, mode)


def kit(directory: str, index: str | None = None, base_model: str | None = None, force: bool = False) -> dict:
    """Write the fine-tuning kit. Runs nothing; a person runs the scripts on the GPU host."""
    from . import evaluation, incubator
    base = base_model or _cfg().get("base_model")
    if not base:
        raise ValueError("no base model — give --base (a Hugging Face id or directory of the model to fine-tune) or config finetune.base_model")
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    manifest = incubator.hatch(str(d / "dataset"), index, force)
    split = _split(d / "dataset" / "qa.jsonl", d / "split")
    qp = evaluation.eval_dir() / "recall.json"
    questions = [{"frage": q["frage"], "stichworte": q["stichworte"]} for q in evaluation.load_questions(qp)] if qp.exists() else []
    _write(d / "eval_questions.json", json.dumps(questions, ensure_ascii=False, indent=1))
    _write(d / "train.py", TRAIN_PY, 0o700)
    _write(d / "training_runtime.py", Path(__file__).with_name("training_runtime.py").read_text(encoding="utf-8"))
    _write(d / "requirements-train.txt", "transformers==5.23.0\ntrl==1.18.0\npeft==0.21.3\ndatasets==5.2.0\naccelerate==1.15.0\n")
    import clingo_object
    from clingo_object import notation
    core = d / "clingo_object"
    core.mkdir(mode=0o700, exist_ok=True)
    for source in sorted(Path(clingo_object.__file__).resolve().parent.glob("*.py")):
        _write(core / source.name, source.read_text(encoding="utf-8"))
    _write(d / "SOURCE-NOTICE.txt", f"Bundled clingo-object {clingo_object.__version__}; source: https://github.com/imoes/vi-mcp\n"
           "SPDX-License-Identifier: AGPL-3.0-or-later\nLicense: https://github.com/imoes/vi-mcp/blob/main/LICENSE\n")
    with (d / "requirements-train.txt").open("a", encoding="utf-8") as f:
        f.write(f"clingo>={notation.CLINGO_MIN}\n")
    _write(d / "evaluate.py", EVALUATE_PY, 0o700)
    _write(d / "evaluation_core.py", Path(__file__).with_name("training_eval.py").read_text(encoding="utf-8"))
    cases, seen = [], set()
    for stage in ("logic", "kto"):
        for line in (d / "split" / f"{stage}.test.jsonl").read_text().splitlines():
            row = json.loads(line)
            m = row.get("meta") or {}
            case = m.get("program_eval")
            if not case:
                continue
            ident = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
            if ident not in seen:
                seen.add(ident)
                cases.append({"id": ident, "group": m["group"], **case, "reference": "explicit reference stored with checked execution"})
    _write(d / "program_eval.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases))
    _write(d / "program_evaluate.py", PROGRAM_EVALUATE_PY, 0o700)
    _write(d / "heldout_loss.py", HELDOUT_LOSS_PY, 0o700)
    _write(d / "synth_qa.py", SYNTH_QA_PY, 0o700)
    _write(d / "convert.sh", CONVERT_SH, 0o700)
    _write(d / "README-FINETUNE.md", GUIDE)
    run = {"base_model": base, "kit": g.now(), "vi_mcp": __version__, "handbook": GUIDE_VERSION, "index": index,
           "dataset_manifest_sha256": hashlib.sha256((d / "dataset" / "manifest.json").read_bytes()).hexdigest(), "split": split,
           "incubator": manifest.get("inkubator"),
           "kit_code_sha256": {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [d / "train.py", d / "training_runtime.py", d / "evaluate.py", d / "program_evaluate.py", d / "heldout_loss.py",
                                         d / "synth_qa.py",
                                         d / "evaluation_core.py", *sorted(core.glob("*.py"))]}}
    _write(d / "run.json", json.dumps(run, indent=2, ensure_ascii=False))
    return {"verzeichnis": str(d), **run, "eval_fragen": len(questions)}


def record(directory: str, who: str = "admin") -> g.MemoryObject:
    """A finished run as a model object with its lineage: dataset manifest, base model, adapter hash, evaluation, incubator state."""
    d = Path(directory)
    if not (d / "run.json").exists():
        raise ValueError(f"{d}: no run.json — not a kit written by vi-mcp finetune kit")
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    adapters = {}
    for stage in ("sft", "agentic", "logic", "kto", "dpo"):
        f = d / "out" / stage / "adapter_model.safetensors"
        if f.exists():
            adapters[stage] = hashlib.sha256(f.read_bytes()).hexdigest()
    ep = d / "program_eval_result.json" if (d / "program_eval_result.json").exists() else d / "eval.json"
    ev = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else None
    if not adapters:
        raise ValueError(f"{d}: no trained adapter under out/ — run train.py first")
    evaluation_path = d / "program_eval.jsonl" if ev and ev.get("evaluation_kind") == "program" else d / "split" / "test.jsonl"
    evaluation_hash = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    better = bool(ev and ev.get("metric") == "verified_task_success" and ev.get("evaluation_set_sha256") == evaluation_hash
                  and ev.get("n", 0) > 0 and ev.get("decision") is True)
    verdict = ("not evaluated — run evaluate.py" if ev is None else
               f"fine-tuned {ev['tuned']} vs base {ev['base']} on {ev['n']} held-out questions ({ev['better']} better, {ev['worse']} worse)")
    oid = f"modell:finetune:{run['kit'][:19].replace(':', '').replace('-', '')}"
    data = {**run, "adapter_sha256": adapters, "bewertung": ev, "urteil": verdict, "erfasst": g.now()}
    text = f"Fine-tuned model on {run['base_model']} ({', '.join(adapters)}): {verdict}"
    o = g.load(oid)
    if o is None:
        return g.new_object("modell", text, ["modell:finetune"], data, importance=2, who=who, oid=oid, mode="aktiv" if better else "test",
                            reason="fine-tuning run recorded", instance=g.SERVER_INSTANCE, index=run.get("index"))
    o.text, o.data, o.mode = text, data, "aktiv" if better else "test"
    o.change(who, "fine-tuning run updated", verdict)
    return g.save(o)


def main(argv: list[str]) -> int:
    """vi-mcp finetune guide | plan [--index X] [--base M] | kit DIR [--index X] [--base M] [--force] | record DIR"""
    import argparse
    from . import config
    a = argparse.ArgumentParser(prog="vi-mcp finetune", description="fine-tune an AI with what this VI learned (vi-mcp runs no training)")
    u = a.add_subparsers(dest="cmd", required=True)
    u.add_parser("guide", help="the handbook: what fine-tuning is, programs, hardware, llama.cpp, measuring")
    p = u.add_parser("plan", help="readiness and the steps for this server")
    k = u.add_parser("kit", help="write the training data, test split and scripts into a directory")
    r = u.add_parser("record", help="record a finished run as a model object with its lineage")
    s = u.add_parser("split", help="redo the kit's shared split after synth_qa.py wrote dataset/qa_synth.jsonl (needs no database)")
    s.add_argument("directory")
    for x in (p, k):
        x.add_argument("--index")
        x.add_argument("--base")
    k.add_argument("directory")
    k.add_argument("--force", action="store_true", help="also for a VI still in the incubator (the datasheet names the open criteria)")
    r.add_argument("directory")
    o = a.parse_args(argv)
    if o.cmd == "guide":
        print(GUIDE)
        return 0
    if o.cmd == "split":
        try:
            print(json.dumps(resplit(o.directory), indent=2, ensure_ascii=False))
        except ValueError as e:
            print(f"vi-mcp finetune: {e}")
            return 1
        return 0
    g.open_db(config.load().db)
    try:
        if o.cmd == "plan":
            print("\n".join(plan_lines(plan(o.index, o.base))))
        elif o.cmd == "kit":
            print(json.dumps(kit(o.directory, o.index, o.base, o.force), indent=2, ensure_ascii=False))
        else:
            print(record(o.directory).text)
    except ValueError as e:
        print(f"vi-mcp finetune: {e}")
        return 1
    finally:
        g.close_db()
    return 0
