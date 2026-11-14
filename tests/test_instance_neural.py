"""Full private instance checkpoint with a real frozen causal transformer and an independent mutable kNN clone."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from clingo_object.runtime import Instance
from vi_mcp.knn_lm import Datastore, KNNLanguageModel, TransformersCore
from vi_mcp.machine_host import KNNProgrammer
from vi_mcp import memory as g
from tests.test_instance import LocalMemory, setup, Device, descriptor


def core(seed=4):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.manual_seed(seed)
    tok = Tokenizer(WordLevel({"[UNK]": 0, "service": 1, "port": 2, "80": 3, "8080": 4, "[EOS]": 5}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=64, n_embd=8, n_layer=1, n_head=2, eos_token_id=5))
    return TransformersCore(model, tok, "instance-test-transformer")


def test_full_instance_boot_preserves_context_policy_program_and_neural_memory(gd, tmp_path):
    neural_core = core()
    lm = KNNLanguageModel(neural_core, Datastore(tmp_path / "live", neural_core.identity), weight=.99, k=1)
    lm.remember("inventory", "service port", " 80")
    programmer = KNNProgrammer(lm)
    devices = {"inventory": Device(descriptor(), lambda _: {"port": 80})}
    instance = setup(programmer, devices, ["inventory"])
    instance.call("context", payload={"messages": [{"role": "system", "content": "Reply in German"}]})
    instance.call("execute", instruction="def.rule port", payload={"source": 'ans("80")'})
    receipt = instance.call("execute", instruction="ask.rule port")
    instance.call("execute", instruction="rate " + receipt["receipt"], payload={"expected_answers": ['ans("80")'],
                  "reference": "independent checkpoint fixture", "question": "Which port was checked?"})
    frozen_evidence = instance.inspect()["evidence"]
    frozen_world = instance.inspect()["world"]
    checkpoint = tmp_path / "bundle"
    instance.checkpoint(checkpoint, "full-instance")
    lm.remember("inventory", "service port", " 8080")
    instance.call("context", payload={"replace": True, "reason": "new experiment", "messages": [{"role": "user", "content": "New context"}]})
    instance.call("execute", instruction="def.rule port", payload={"source": 'ans("8080")'})
    clone_model = KNNProgrammer.boot(neural_core, checkpoint, tmp_path / "clone-memory")
    clone = Instance.boot(LocalMemory(), "clone", clone_model, checkpoint, devices=devices)
    assert clone_model.lm.core is neural_core  # shared immutable computation, independent mutable memory
    assert clone_model.lm.lock is lm.lock  # all callers serialize shared model buffers, not only each individual datastore
    assert clone.inspect()["world"] == frozen_world
    assert clone.inspect()["evidence"] == frozen_evidence
    # checked learning evidence survives boot too; the instance context goes along only when the owner says so (review 14.11.2026)
    with pytest.raises(Exception, match="include_context"):
        clone.call("teach", receipt=receipt["receipt"])
    assert clone.call("teach", receipt=receipt["receipt"], include_context=True)["correct"]
    assert lm.predict("service port")["token"] == 4
    assert clone_model.lm.predict("service port")["token"] == 3
    g.close_db()
    g.open_db(gd)
    assert clone.inspect()["world"] == frozen_world
    assert instance.inspect()["world"]["rules"]["port"] == 'ans("8080")'
    with pytest.raises(ValueError, match="mismatch"):
        KNNProgrammer.boot(core(seed=7), checkpoint, tmp_path / "wrong-core")
    data = json.loads((checkpoint / "instance.json").read_text())
    data["world"]["context"] = []
    (checkpoint / "instance.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corrupt"):
        Instance.boot(LocalMemory(), "bad", clone_model, checkpoint, devices=devices)


def test_neural_token_memory_writes_executable_clingo_program_without_weight_training(gd, tmp_path):
    pytest.importorskip("clingo")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    # Fixed byte vocabulary, a real causal transformer, no tokenizer fitting and no parameter updates.
    vocabulary = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update({"[EOS]": 256, "[UNK]": 257})
    tokenizer = Tokenizer(BPE(vocabulary, [], unk_token="[UNK]"))
    tokenizer.pre_tokenizer, tokenizer.decoder = ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="[EOS]", unk_token="[UNK]")
    torch.manual_seed(5)
    neural_core = TransformersCore(GPT2LMHeadModel(GPT2Config(vocab_size=258, n_positions=256, n_embd=32, n_layer=1, n_head=2,
                                       eos_token_id=256)), tokenizer, "byte-transformer-program-contract")
    lm = KNNLanguageModel(neural_core, Datastore(tmp_path / "program-memory", neural_core.identity), weight=1, k=1)
    program = 'ans(P) <= (O in objects("service:api")) & (O.port == P)'
    prompt = "Write the API port query: "
    before = lm.generate(prompt, steps=len(program) + 1)
    assert before["text"] != program
    weights = neural_core.weight_hash()
    lm.remember("independently-checked-program", prompt, program + "[EOS]")
    learned = lm.generate(prompt, steps=len(program) + 1)
    assert learned["finish_reason"] == "stop" and learned["text"] == program
    assert neural_core.weight_hash() == weights
    instance = setup(KNNProgrammer(lm))
    for port in (80, 8080):
        instance.call("execute", instruction="tell.fact api", payload={"text": "API", "source": "independent fixture",
                      "hashtags": ["service:api", f"port:{port}"]})
        result = instance.call("execute", instruction="ask.rule", payload={"source": learned["text"]})
        assert result["answers"] == [f'ans("{port}")']
