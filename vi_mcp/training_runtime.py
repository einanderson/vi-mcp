"""Standalone training-kit runtime: audited tokenization, assistant targets and real trainer preflight without weight updates.

Copied into a fine-tuning kit; never imports the VI server or runs training on import. Optional ML dependencies load only in main.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

FILES = {"sft": "train.jsonl", "agentic": "agentic.train.jsonl", "logic": "logic.train.jsonl",
         "kto": "kto.train.jsonl", "dpo": "dpo.train.jsonl"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def versions():
    found = {}
    for package in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = "not installed"
    return found


def write_report(path, report):
    with Path(path).open("w", encoding="utf-8") as f:
        os.chmod(f.name, 0o600)
        json.dump(report, f, indent=2, ensure_ascii=False)


def encode(tokenizer, text, *, special=False):
    return list(tokenizer(text, add_special_tokens=special, truncation=False)["input_ids"])


def prepare(rows, tokenizer, stage, limit, *, skip_oversize=False):
    """Render original JSON before Arrow can turn distinct tool-argument dictionaries into a union full of null fields.
    Each SFT assistant turn is a complete target; user text, old assistant turns and tool observations are masked.
    Preference strings keep their defined format; all token boundaries and later trainer output are checked exactly.
    """
    if type(limit) is not int or limit < 2:
        raise ValueError("sequence limit must be at least two tokens")
    prepared, audit = [], {"rows": len(rows), "examples": 0, "target_tokens": 0, "longest_sequence": 0, "rejected": {}, "skip_oversize": skip_oversize}
    rejected = Counter()
    for row in rows:
        try:
            candidates = []
            if stage in ("sft", "agentic", "logic"):
                messages = row["messages"]
                if not isinstance(messages, list) or not messages or not tokenizer.chat_template:
                    raise ValueError("conversation_or_template")
                for i, message in enumerate(messages):
                    if message.get("role") != "assistant":
                        continue
                    if not message.get("content") and not message.get("tool_calls"):
                        continue
                    kwargs = {"enable_thinking": False, "tokenize": False, **({"tools": row["tools"]} if row.get("tools") is not None else {})}
                    prompt = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, **kwargs)
                    full = tokenizer.apply_chat_template(messages[:i + 1], add_generation_prompt=False, **kwargs)
                    prefix, ids = encode(tokenizer, prompt), encode(tokenizer, full)
                    if ids[:len(prefix)] != prefix:
                        raise ValueError("template_prefix_mismatch")
                    if len(ids) <= len(prefix):
                        raise ValueError("empty_target")
                    candidates.append(({"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix):]}, len(ids), len(ids) - len(prefix)))
                if not candidates:
                    raise ValueError("no_assistant_target")
            else:
                prompt = row["prompt"]
                fields = ("completion",) if stage == "kto" else ("chosen", "rejected")
                if not isinstance(prompt, str) or not prompt or any(not isinstance(row[k], str) or not row[k] for k in fields):
                    raise ValueError("preference_format")
                if tokenizer.chat_template:
                    # TRL applies no chat template to string rows; render it here like the SFT stages (review 14.11.2026, agreed
                    # by Codex), so preference learning sees the format of inference. The audit below then checks these tokens.
                    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                           enable_thinking=False, tokenize=False)
                    bos = getattr(tokenizer, "bos_token", None)
                    if bos and prompt.startswith(bos) and encode(tokenizer, prompt, special=True)[:2] == encode(tokenizer, bos + bos, special=False)[:2]:
                        raise ValueError("double_bos")
                if stage == "kto" and type(row.get("label")) is not bool:
                    raise ValueError("binary_label")
                if stage == "dpo" and row["chosen"] == row["rejected"]:
                    raise ValueError("identical_preference")
                item = {"prompt": prompt, **{k: row[k] for k in fields}, **({"label": row["label"]} if stage == "kto" else {})}
                prefix = encode(tokenizer, prompt, special=True)
                sizes, targets = [], 0
                for field in fields:
                    completion = row[field]
                    if not tokenizer.eos_token:
                        raise ValueError("missing_eos")
                    if not completion.endswith(tokenizer.eos_token):
                        completion += tokenizer.eos_token
                    ids = encode(tokenizer, prompt + completion, special=True)
                    if ids[:len(prefix)] != prefix:
                        raise ValueError("preference_prefix_mismatch")
                    if len(ids) <= len(prefix):
                        raise ValueError("empty_target")
                    sizes.append(len(ids))
                    targets += len(ids) - len(prefix)
                candidates.append((item, max(sizes), targets))
            # An episode remains complete: do not keep only its shorter assistant turn when a later answer is too large.
            longest = max(size for _item, size, _n in candidates)
            audit["longest_sequence"] = max(audit["longest_sequence"], longest)
            if longest > limit:
                rejected["oversize"] += 1
                continue
            for item, _size, n in candidates:
                prepared.append(item)
                audit["target_tokens"] += n
        except (ValueError, KeyError, TypeError) as e:
            allowed = {"conversation_or_template", "template_prefix_mismatch", "empty_target", "no_assistant_target",
                       "preference_format", "binary_label", "identical_preference", "missing_eos", "preference_prefix_mismatch", "double_bos"}
            rejected[str(e) if str(e) in allowed else "invalid_example"] += 1
        except Exception:
            rejected["chat_template_error"] += 1
    audit.update(examples=len(prepared), rejected=dict(rejected))
    if stage in ("kto", "dpo"):
        audit["preference_format"] = "chat_template" if tokenizer.chat_template else "plain (tokenizer has no chat template)"
    audit["valid"] = bool(prepared) and not any(n for key, n in rejected.items() if key != "oversize" or not skip_oversize)
    return prepared, audit


def verify_prepared(trainer, rows, tokenizer, stage, limit):
    """No silent library truncation, column loss or changed targets. KL mismatches must fit the model too."""
    data = trainer.train_dataset
    if len(data) != len(rows):
        raise ValueError("trainer changed the number of examples")
    for expected, actual in zip(rows, data):
        if stage in ("sft", "agentic", "logic"):
            if actual["input_ids"] != expected["input_ids"] or actual["labels"] != expected["labels"]:
                raise ValueError("trainer changed audited assistant tokens or masks")
        else:
            prefix = encode(tokenizer, expected["prompt"], special=True)
            if actual["prompt_ids"] != prefix:
                raise ValueError("trainer changed the preference prompt")
            fields = {"completion": "completion_ids"} if stage == "kto" else {"chosen": "chosen_ids", "rejected": "rejected_ids"}
            for field, column in fields.items():
                text = expected[field]
                if not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token
                ids = encode(tokenizer, expected["prompt"] + text, special=True)
                if actual[column] != ids[len(prefix):]:
                    raise ValueError("trainer changed preference target tokens")
            if stage == "kto" and "KL_completion_ids" in actual and len(prefix) + len(actual["KL_completion_ids"]) > limit:
                raise ValueError("KTO mismatched reference exceeds the model context")


def parameter_hash(model):
    import torch
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(str((p.shape, p.dtype)).encode())
        h.update(memoryview(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return h.hexdigest()


def verify_batch(batch, expected, tokenizer, stage, limit):
    allowed = set()
    for row in expected:
        if stage in ("sft", "agentic", "logic"):
            allowed.add((tuple(row["input_ids"]), tuple(row["labels"])))
        else:
            prefix = encode(tokenizer, row["prompt"], special=True)
            for field in (("completion",) if stage == "kto" else ("chosen", "rejected")):
                text = row[field] if row[field].endswith(tokenizer.eos_token) else row[field] + tokenizer.eos_token
                ids = encode(tokenizer, row["prompt"] + text, special=True)
                allowed.add((tuple(ids), tuple([0] * len(prefix) + [1] * (len(ids) - len(prefix)))))
    target_key = "labels" if stage in ("sft", "agentic", "logic") else "completion_mask"
    for ids, active, target in zip(batch["input_ids"].tolist(), batch["attention_mask"].tolist(), batch[target_key].tolist()):
        ids = tuple(v for v, keep in zip(ids, active) if keep)
        target = tuple(v for v, keep in zip(target, active) if keep)
        if len(ids) > limit or (ids, target) not in allowed:
            raise ValueError("collator changed audited tokens or target masks")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=list(FILES), default="sft")
    parser.add_argument("--base")
    parser.add_argument("--revision", help="pin a remote model/tokenizer commit")
    parser.add_argument("--from-adapter", default="out/sft")
    parser.add_argument("--from-base", action="store_true", help="start a new adapter for this stage directly from the base, without a prior QA stage")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true", help="construct trainer, collate and compute one loss; no train/save/optimizer step")
    parser.add_argument("--tokenizer-only", action="store_true", help="preflight tokenization only; no model weights or trainer loaded")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sequence-length", type=int, help="audited full-example bound, at most the model context")
    parser.add_argument("--skip-oversize", action="store_true", help="explicitly omit whole oversized episodes and count them; never truncate targets")
    o = parser.parse_args(argv)
    o.preflight = o.preflight or o.tokenizer_only
    report_path = Path(f"preflight-{o.stage}.json" if o.preflight else f"preparation-{o.stage}.json")
    report = {"stage": o.stage, "training_started": False, "optimizer_steps": 0, "status": "failed", "trainer_verified": False}
    trainer = None
    try:
        run = json.loads(Path("run.json").read_text())
        base = o.base or run["base_model"]
        if base != run["base_model"] and not o.preflight:
            raise ValueError("create a new kit to train a different base model")
        source = Path("split") / FILES[o.stage]
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        report.update(dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), base_model=base, kit_base_model=run["base_model"],
                      probe_model_override=base != run["base_model"], versions=versions())
        from transformers import AutoConfig, AutoTokenizer
        parent = Path(o.from_adapter)
        if o.from_base and o.from_adapter != "out/sft":
            raise ValueError("choose either a parent adapter or a fresh base")
        parent_config = (json.loads((parent / "adapter_config.json").read_text())
                         if o.stage != "sft" and not o.from_base and (parent / "adapter_config.json").exists() else None)
        if parent_config is not None:
            if parent_config["base_model_name_or_path"] != base:
                raise ValueError("parent adapter belongs to a different base model")
            if o.revision is not None and o.revision != parent_config.get("revision"):
                raise ValueError("requested revision differs from the parent adapter's recorded base revision")
        revision = parent_config.get("revision") if parent_config is not None else o.revision
        tokenizer_source = parent if o.stage != "sft" and not o.from_base and (parent / "tokenizer_config.json").exists() else base
        tok = AutoTokenizer.from_pretrained(tokenizer_source, revision=revision if tokenizer_source == base else None, trust_remote_code=False)
        config = AutoConfig.from_pretrained(base, revision=revision, trust_remote_code=False)
        window = getattr(config, "max_position_embeddings", getattr(config, "n_positions", None))
        if type(window) is not int or window < 2:
            raise ValueError("model must declare its context window")
        limit = o.sequence_length if o.sequence_length is not None else window
        if type(limit) is not int or not 2 <= limit <= window:
            raise ValueError("sequence limit exceeds the declared model context")
        data, audit = prepare(rows, tok, o.stage, limit, skip_oversize=o.skip_oversize)
        report.update(audit=audit, sequence_limit=limit, model_revision=getattr(config, "_commit_hash", None) or revision,
                      tokenizer_sha256=digest(tok.backend_tokenizer.to_str()),
                      chat_template_sha256=digest(tok.chat_template), model_config_sha256=digest(config.to_dict()))
        if not audit["valid"]:
            raise ValueError("training examples failed the token audit; inspect rejection classes")
        if o.tokenizer_only:
            report.update(status="tokenizer_passed", scope="tokenization only; trainer and hardware not verified")
            write_report(report_path, report)
            print(json.dumps(report, indent=2))
            return 0
        import torch
        from datasets import Dataset
        from peft import LoraConfig, AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        o.cpu = o.cpu or (o.preflight and not torch.cuda.is_available())
        torch.manual_seed(o.seed)
        if o.cpu and o.qlora:
            raise ValueError("this kit's 4-bit training path requires an appropriate accelerator")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if o.qlora else None
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules="all-linear", task_type="CAUSAL_LM",
                          revision=report["model_revision"])
        if parent_config is not None:
            model = AutoPeftModelForCausalLM.from_pretrained(parent, is_trainable=True, quantization_config=quant,
                                                            dtype=torch.float32 if o.cpu else "auto", trust_remote_code=False)
            peft, report["parent_adapter"] = None, str(parent)
        else:
            if o.stage != "sft" and not o.from_base and not o.preflight:
                raise ValueError("later stages require an existing parent adapter")
            model = AutoModelForCausalLM.from_pretrained(base, revision=report["model_revision"], quantization_config=quant,
                                                         dtype=torch.float32 if o.cpu else "auto", trust_remote_code=False)
            peft = lora
            report["parent_adapter"] = "base" if o.stage == "sft" or o.from_base else "missing: fresh untrained probe only"
        from trl import SFTConfig, SFTTrainer, KTOConfig, KTOTrainer, DPOConfig, DPOTrainer
        classes = {"kto": (KTOConfig, KTOTrainer), "dpo": (DPOConfig, DPOTrainer)}
        cfg_class, trainer_class = classes.get(o.stage, (SFTConfig, SFTTrainer))
        options = {"output_dir": str(Path("out") / o.stage), "num_train_epochs": o.epochs,
                   "learning_rate": o.lr or (2e-4 if o.stage == "sft" else 1e-4 if o.stage in ("agentic", "logic") else 5e-6),
                   "logging_steps": 10, "max_length": None, "report_to": "none", "use_cpu": o.cpu, "seed": o.seed,
                   "bf16": not o.cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "fp16": False,
                   "per_device_train_batch_size": 4 if o.stage == "kto" else 1}
        if o.cpu:
            options.update(bf16=False, fp16=False)
        if o.preflight:
            checkpointing = "activation_checkpointing" if "activation_checkpointing" in cfg_class.__dataclass_fields__ else "gradient_checkpointing"
            options[checkpointing] = False
        if o.stage in ("sft", "agentic", "logic"):
            options.update(dataset_kwargs={"skip_prepare_dataset": True}, packing=False)
        trainer = trainer_class(model=model, processing_class=tok, train_dataset=Dataset.from_list(data),
                                peft_config=peft, args=cfg_class(**options))
        verify_prepared(trainer, data, tok, o.stage, limit)
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        if not trainable:
            raise ValueError("the adapter has no trainable parameters")
        report.update(trainer_verified=True, parent_ready=report["parent_adapter"] != "missing: fresh untrained probe only",
                      incubator_mature=(run.get("incubator") or {}).get("zustand") == "reif", trainable_parameters=trainable, seed=o.seed)
        if o.preflight:
            before = parameter_hash(trainer.model)
            batch = next(iter(trainer.get_train_dataloader()))
            verify_batch(batch, data, tok, o.stage, limit)
            trainer.model.eval()
            with torch.no_grad():
                loss = trainer.compute_loss(trainer.model, batch)
            after = parameter_hash(trainer.model)
            if before != after or not math.isfinite(float(loss.detach())) or trainer.state.global_step != 0 or trainer.optimizer is not None:
                raise ValueError("preflight changed parameters or produced a nonfinite loss")
            report.update(status="preflight_passed", parameter_sha256=before, parameters_unchanged=True,
                          loss=float(loss.detach()), batch_shapes={k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
            write_report(report_path, report)
            print(json.dumps(report, indent=2))
            return 0
        report["status"], report["training_started"], report["optimizer_steps"] = "prepared", True, None
        write_report(report_path, report)
        trainer.train()
        trainer.save_model(str(Path("out") / o.stage))
        report.update(status="trained", optimizer_steps=int(trainer.state.global_step))
        write_report(report_path, report)
        run.update({f"trained_{o.stage}": str(Path("out") / o.stage), f"parent_{o.stage}": base if o.stage == "sft" or o.from_base else str(parent),
                    f"preparation_{o.stage}_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()})
        write_report("run.json", run)
        return 0
    except Exception as e:
        report.update(status="failed", error_class=type(e).__name__)
        if trainer is not None:
            report["optimizer_steps"] = int(trainer.state.global_step)
        write_report(report_path, report)
        print(f"Preparation refused ({type(e).__name__}); see {report_path}")
        return 2
