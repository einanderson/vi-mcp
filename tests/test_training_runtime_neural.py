"""Actual exported kits through real Transformers/PEFT/TRL, without train(), backward() or optimizer steps."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("clingo")

from vi_mcp import evaluation, finetune, logic, machine
from vi_mcp import memory as g


@pytest.fixture()
def kit(gd, tmp_path, monkeypatch):
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    vocabulary = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update({"[EOS]": 256, "[UNK]": 257, "[PAD]": 258})
    tokenizer = Tokenizer(BPE(vocabulary, [], unk_token="[UNK]"))
    tokenizer.pre_tokenizer, tokenizer.decoder = ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]")
    tok.chat_template = ("{% if tools %}{{ tools | tojson }}{% endif %}{% for m in messages %}<{{m.role}}>"
                         "{{m.content or ''}}{% if m.tool_calls %}{{m.tool_calls | tojson}}{% endif %}{{eos_token}}{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
    base = tmp_path / "base"
    tok.save_pretrained(base)
    torch.manual_seed(2)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=16384, n_embd=8, n_layer=1, n_head=2,
                             eos_token_id=256, pad_token_id=258)).save_pretrained(base)
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    for i in range(20):
        g.new_object("erkenntnis", f"service{i} deployment", [f"thema:service{i}"],
            {"quelle": f"fixture-page-{i}", "inhalt": f"service{i} uses port {8000+i}", "klasse": {"fragen": [f"How to deploy service{i}?"]}},
            visibility="intern", oid=f"erkenntnis:fixture-{i}")
    def call(action, **kw):
        return machine.call("preflight", {"action": action, "name": "test", **kw})
    call("create")
    call("execute", instruction="tell.fact api", payload={"text": "API", "source": "independent fixture", "hashtags": ["service:api", "port:8080"]})
    for source in ('ans(P) <= (O in objects("service:api")) & (O.port == P)', 'ans("wrong")'):
        result = call("execute", instruction="ask.rule", payload={"source": source})
        call("execute", instruction="rate " + result["receipt"], payload={"expected_answers": ['ans("8080")'],
             "reference": "independent inventory fixture", "question": "Which port is the API using?"})
        call("teach", receipt=result["receipt"])
    logic.verify('ans("safe")', ['ans("safe")'], "independent predicate fixture", "What is safe?", "preflight")
    directory = tmp_path / "kit"
    finetune.kit(str(directory), base_model=str(base), force=True)
    return directory, base


@pytest.mark.parametrize("stage", ["sft", "agentic", "logic", "kto", "dpo"])
def test_generated_script_preflight_uses_real_trainers_but_never_trains(kit, monkeypatch, stage):
    from trl import SFTTrainer, KTOTrainer, DPOTrainer
    directory, base = kit
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not train, save model weights or run an optimizer")
    for trainer in (SFTTrainer, KTOTrainer, DPOTrainer):
        monkeypatch.setattr(trainer, "train", forbidden)
        monkeypatch.setattr(trainer, "save_model", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    before = hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()
    run = (directory / "run.json").read_bytes()
    monkeypatch.chdir(directory)
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, "training_runtime", raising=False)
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage", stage, "--preflight", "--cpu", "--from-base"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(directory / "train.py"), run_name="__main__")
    assert exited.value.code == 0
    report = json.loads((directory / f"preflight-{stage}.json").read_text())
    assert report["status"] == "preflight_passed" and report["trainer_verified"]
    assert report["parameters_unchanged"] and report["optimizer_steps"] == 0 and not report["training_started"]
    assert report["audit"]["target_tokens"] and not report["audit"]["rejected"]
    assert report["trainable_parameters"] > 0 and report["parent_ready"]
    assert not report["incubator_mature"]  # a successful API check must not pretend that the VI is ready
    assert hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest() == before
    assert not list((directory / "out").glob("*/adapter_model.*"))
    assert (directory / "run.json").read_bytes() == run


def test_tokenizer_only_and_oversize_gate_load_no_model_weights(kit, monkeypatch):
    from vi_mcp.training_runtime import main
    from transformers import AutoModelForCausalLM
    directory, _base = kit
    monkeypatch.chdir(directory)
    def forbidden(*args, **kwargs):
        raise AssertionError("token-only audit must not load model weights")
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    assert main(["--stage", "logic", "--tokenizer-only"]) == 0
    report = json.loads((directory / "preflight-logic.json").read_text())
    assert report["status"] == "tokenizer_passed" and not report["trainer_verified"]
    assert report["precision"].startswith("model dtype")
    assert main(["--stage", "logic", "--tokenizer-only", "--fp32"]) == 0          # gfx1151: bf16 problems reported
    assert json.loads((directory / "preflight-logic.json").read_text())["precision"] == "float32"
    assert main(["--stage", "logic", "--preflight", "--sequence-length", "16"]) == 2
    report = json.loads((directory / "preflight-logic.json").read_text())
    assert report["error_class"] == "ValueError" and report["audit"]["rejected"]["oversize"]


def test_a_later_stage_can_load_a_valid_parent_adapter_without_training_it(kit, monkeypatch, tmp_path):
    from peft import get_peft_model, LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vi_mcp.training_runtime import main
    directory, base = kit
    # An initialized, untrained fixture verifies the adapter file/interface path; it is never recorded as a trained result.
    parent = tmp_path / "initialized-parent"
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(str(base)), LoraConfig(r=2, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.save_pretrained(parent)
    AutoTokenizer.from_pretrained(base).save_pretrained(parent)
    before = (parent / "adapter_model.safetensors").read_bytes()
    monkeypatch.chdir(directory)
    assert main(["--stage", "logic", "--preflight", "--cpu", "--from-adapter", str(parent)]) == 0
    report = json.loads((directory / "preflight-logic.json").read_text())
    assert report["parent_adapter"] == str(parent) and report["parent_ready"] and report["parameters_unchanged"]
    assert report["optimizer_steps"] == 0 and (parent / "adapter_model.safetensors").read_bytes() == before
    # A later stage must not silently load different base weights than its adapter's recorded revision.
    config = json.loads((parent / "adapter_config.json").read_text())
    config["revision"] = "a" * 40
    (parent / "adapter_config.json").write_text(json.dumps(config))
    assert main(["--stage", "logic", "--preflight", "--cpu", "--from-adapter", str(parent), "--revision", "b" * 40]) == 2
    report = json.loads((directory / "preflight-logic.json").read_text())
    assert report["error_class"] == "ValueError" and not report["trainer_verified"] and report["optimizer_steps"] == 0


def test_heldout_loss_of_a_fresh_adapter_equals_the_base(kit, monkeypatch):
    """heldout_loss.py: same tokens/masks as training on the held-out split; a fresh LoRA adapter (B = 0) must give exactly the base
    loss — the measurement compares like with like."""
    import subprocess, sys
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    directory, base = kit
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base), LoraConfig(r=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    model.save_pretrained(directory / "out" / "sft")
    r = subprocess.run([sys.executable, "heldout_loss.py", "--stage", "sft", "--examples", "5"], cwd=directory, capture_output=True,
                       text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads((directory / "heldout-loss-sft.json").read_text())
    assert out["examples"] > 0 and math.isfinite(out["base_nll"]) and abs(out["difference"]) < 1e-6
